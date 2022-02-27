import sys
import os

from transformers.models import bert
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter

# from sentence_transformers import SentenceTransformer
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件夹

def get_subj_obj_start(input_ids_arr:list,tokenizer,additional_index):
    """
    input_ids_arr like:
        tensor([[  101,  9499,  1071,  2149, 30522,  8696, 30522, 30534,  6874,  9033,
            4877,  3762, 30534, 10650,  1999, 12867,  1024,  5160,   102,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0],
            [  101,  2019, 21931, 17680,  2013, 11587, 30532,  2149, 30532, 14344,
            5016, 30537,  2406, 22517,  3361, 30537,  2006,  5958,  1010, 11211,
            2007, 10908,  2005,  1037,  2149,  3446,  3013,  2006,  9317,  1010,
            2992,  8069,  2008,  1996, 23902,  2013,  1996,  2149,  3847, 24185,
            2229,  2003, 24070,  1010, 16743,  2056,  1012,   102]])
    tokenizer:
        as named
    additional_index:
        the first index of the additional_special_tokens
    return:
         subj and obj start position
    """
    subj_starts = []
    obj_starts = []
    for input_ids in input_ids_arr:
        subj_start = -1
        obj_start = -1
        checked_id = []
        for idx,word_id in enumerate(input_ids):
            if subj_start!=-1 and obj_start!=-1:
                break
            if word_id>=additional_index:
                if word_id not in checked_id:
                    checked_id.append(word_id)
                    decoded_word = tokenizer.decode(word_id)
                    if decoded_word.startswith("<S:"):
                        subj_start = idx
                    elif decoded_word.startswith("<O:"):
                        obj_start = idx
        if subj_start==-1 or obj_start==-1:
            # print(tokenizer.batch_decode(input_ids))
            if subj_start==-1:
                subj_start=0
            if obj_start==-1:
                obj_start=0
        subj_starts.append(subj_start)
        obj_starts.append(obj_start)
    return subj_starts,obj_starts

class SCCL_BERT(nn.Module):

    def __init__(self,bert_model,max_length,device,n_rel,open_bert = True,e_tags = []):
        super(SCCL_BERT, self).__init__()
        print("SCCL_BERT init")
        self.device = device
        #self.pcnn = pcnn_encoder
        self.max_length = max_length
        self.open_bert = open_bert
        
        # self.alpha = alpha
        #Instance-CL head  embed_dim=>128
        
        self.tokenizer = bert_model[0].tokenizer
        self.sentbert = bert_model[0].auto_model
        self.additional_index = len(self.tokenizer)
        # add special tokens
        if len(e_tags)!=0:
            print("Add {num} special tokens".format(num=len(e_tags)))
            special_tokens_dict = {'additional_special_tokens': e_tags}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.sentbert.resize_token_embeddings(len(self.tokenizer))  # enlarge vocab
        
        self.embed_dim = self.sentbert.config.hidden_size
        #如果不放开bert的话就冻住
        if open_bert==False:
            for param in self.sentbert.parameters():
                param.requires_grad = False
            self.sentbert.eval()
        # instance CL loss 用的g
        # self.head = nn.Sequential(  
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.embed_dim, 128))
        #clustering loss 用的linear
        # self.f = nn.Linear(self.embed_dim,self.embed_dim)
        #帮助分类的全连接层
        self.out = nn.Linear(2*self.embed_dim,n_rel)
        # self.out = nn.Linear(self.embed_dim,n_rel)

        # cluster_centers
        # initial_cluster_centers = torch.tensor(cluster_centers, dtype=torch.float, requires_grad=True)
        # self.cluster_centers = Parameter(initial_cluster_centers) # 优化目标
    @staticmethod
    def cls_pooling(model_output):
        return model_output[0][:,0]  # model_output[0] 表示 last hidden state, bert_output[0].shape => [bs,max_len,d_model]
    def get_embeddings(self, text_arr):
        """
        text_arr: e.g. ["sentence 1","sentence 2"]
        """
        #这里的x都是文本
        feat_text= self.tokenizer.batch_encode_plus(text_arr, 
                                                    max_length=self.max_length+2,  # +2是因为CLS 和SEQ也算进去max_length的
                                                    return_tensors='pt', 
                                                    padding='longest',
                                                    truncation=True)
        #feature的value都放到device中
        for k,_ in feat_text.items():
            feat_text[k] = feat_text[k].to(self.device)
        self.sentbert.train()
        bert_output = self.sentbert.forward(**feat_text)

        #计算embedding (CLS output)
        embedding = SCCL_BERT.cls_pooling(bert_output)
        

        # # sccl 里面的做法
        # attention_mask = feat_text['attention_mask'].unsqueeze(-1)
        # all_output = bert_output[0]  # bert_output is a type of BaseModelOutput
        # embedding1 = torch.sum(all_output * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        # print(embedding1==embedding)

        return embedding
    def get_embeddings_PURE(self,text_arr):
        """
        from paper:
            A Frustratingly Easy Approach for Entity and Relation Extraction
            ent1_spos 是每个句子的entity1的开始位置
            ent2_spos 是每个句子的entity2的开始位置
        """
        feat_text= self.tokenizer.batch_encode_plus(text_arr, 
                                                    max_length=self.max_length+2,  # +2是因为CLS 和SEQ也算进去max_length的
                                                    return_tensors='pt', 
                                                    padding='longest',
                                                    truncation=True)
        #feature的value都放到device中
        for k,_ in feat_text.items():
            feat_text[k] = feat_text[k].to(self.device)
        self.sentbert.train()


        ent1_spos,ent2_spos = get_subj_obj_start(feat_text['input_ids'],self.tokenizer,self.additional_index)

        bert_output = self.sentbert.forward(**feat_text)
        bert_output = bert_output[0]
        bs = bert_output.shape[0]
        assert len(ent1_spos)==len(ent2_spos)
        ent1_spos = torch.tensor(ent1_spos)
        ent2_spos = torch.tensor(ent2_spos)
        embedding1 = bert_output[[i for i in range(bs)],ent1_spos,:]
        embedding2 = bert_output[[i for i in range(bs)],ent2_spos,:]
        embeddings = torch.cat([embedding1,embedding2],dim = 1)
        return embeddings  # [bs, 2*max_len, d_model]



    # def forward(self,texts):
    #     return self.out(self.get_embeddings(texts))

class BertClassifier(nn.Module):
    def __init__(self,tokenizer,model,max_length,device,e_tags:list=None):
        super(BertClassifier, self).__init__()
        self.device = device
        #self.pcnn = pcnn_encoder
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.sentbert = model
        self.max_length = max_length
        if e_tags is not None:
            special_tokens_dict = {'additional_special_tokens': e_tags}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.sentbert.resize_token_embeddings(len(self.tokenizer))  # enlarge vocab
    def classify(self, text_arr:list):
        """
        text_arr:
            input_text, list 
        output:
            [len(text_arr), n_classification]
        """
        feat_text= self.tokenizer.batch_encode_plus(text_arr, 
                                                    max_length=self.max_length+2, 
                                                    return_tensors='pt', 
                                                    padding='longest',
                                                    truncation=True)
        for k,v in feat_text.items():
            feat_text[k] = feat_text[k].to(self.device)
        self.sentbert.train()
        bert_output = self.sentbert.forward(**feat_text)
        return bert_output['logits']
    def forward(self,text_arr):
        output = self.classify(text_arr)
        return output

if __name__=="__main__":
    
    debug_stop = 1

    