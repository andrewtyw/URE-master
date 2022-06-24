import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())

sys.path.append(CURR_DIR)
P = PATH.parent
print("current dir: ",CURR_DIR)
for i in range(1):  
    P = P.parent
    PROJECT_PATH = str(P.absolute())
    sys.path.append(str(P.absolute()))
import time
TIME=time.strftime("%m%d%H%M%S", time.localtime())
print("time",TIME)

from torch.utils.data import Dataset, dataset
import torch.utils.data as util_data
from tqdm import tqdm
import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer
from model.sccl import SCCL_BERT
from utils.dict_relate import dict_index
from utils.randomness import set_global_random_seed
from utils.clean import get_format_train_text, find_uppercase
from utils.pickle_picky import load,save
from  torch.utils import data
import torch.nn.functional as F
import copy
import random
import numpy as np
import argparse
from tqdm import tqdm
from collections import Counter


def cal_acc(pre,true,select_index=None):
    right = 0
    L1 = len(pre)
    L2 = len(true)
    pre = np.array(pre)
    true = np.array(true)
    assert L1==L2
    if select_index is not None:
        assert max(select_index)<=L1-1
        select_index = np.array(select_index)
        pre = pre[select_index]
        true = true[select_index]
        return sum(pre==true)/len(select_index)
    return sum(pre==true)/L1


class Train_dataset(data.Dataset):
    def __init__(self, dataset):
           self.data = dataset
    def __getitem__(self, index):
        return {
            'text':self.data['text'][index],
            'p_label':self.data['p_label'][index],
            'noise_or_not':self.data['noise_or_not'][index],
            'index':self.data['index'][index],
        }
    def __len__(self):
        return len(self.data['text'])

class mydataset(Dataset):  
    def __init__(self, data) -> object:
        assert len(data['text'])==len(data['label'])
        self.text = data['text']
        self.label = torch.tensor(data['label']).long()
        

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {
            'text': self.text[idx],
            'label': self.label[idx]}

class surpervise_learner(nn.Module):
    def __init__(self,args,sccl_bert:SCCL_BERT,optimizer):
        super(surpervise_learner,self).__init__()
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.cuda_index))
        self.CEloss = nn.CrossEntropyLoss(reduction='none')
        self.sccl_bert = sccl_bert
        self.optimizer = optimizer
    def forward(self,p_rel,text_arr):
        p_rel = p_rel.to(self.device)
        embd0 = self.sccl_bert.get_embeddings_PURE(text_arr)
        out = self.sccl_bert.out(embd0)
        loss = self.CEloss(out,p_rel)  
        index_loss = loss.clone()
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        index_loss = index_loss.detach()
        logits = F.softmax(out.detach().cpu(),-1)
        return loss.detach(),index_loss,logits




def adjust_learning_rate(optimizer,epoch,max_epoch,scale):
    if epoch<0.25*max_epoch:
        lr = (5e-6)*3
    elif epoch < 0.5 * max_epoch:
        lr = (5e-6)*2
    else:
        lr = 5e-6
    
    
    
    
    
        
    
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr*scale
    return lr
def first_stage(args,sccl_model:SCCL_BERT,train_loader,train_data,data_for_select):

    ndata=len(train_data['text'])  
    optimizer = torch.optim.Adam([
            {'params':sccl_model.sentbert.parameters()},
            {'params':sccl_model.out.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)
    sup_learner = surpervise_learner(args,sccl_model,optimizer).to(args.device)
    base_test = -1 


    print()
    N_train = len(train_data['text'])
    for epoch in range(1, args.n_epoch+1): 
        globals_loss = 0
        sccl_model.train()
        data_predict = torch.zeros(N_train,dtype=torch.long)
        data_predict_confidence = torch.zeros(N_train)
        lr=adjust_learning_rate(optimizer,epoch,args.n_epoch,args.lr_scale) 
        for i, batch in tqdm(enumerate(train_loader),total=len(train_loader)):
            loss,index_loss,logits = sup_learner.forward(p_rel=batch['p_label'],text_arr=batch['text'])
            globals_loss += index_loss.sum().cpu().data.item()

            
            
            _, pred = torch.max(logits.data, -1)  
            confidences = logits
            
            confidence = confidences[np.array([i for i in range(len(logits))]),pred]
            data_predict[batch['index']] = pred.detach().cpu()
            data_predict_confidence[batch['index']] = confidence.detach().cpu()
            

        print ("epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss /ndata, "test_accuarcy:%f" % base_test)
        

        
        
        

        predicts = dict()
        for i,(pre,confi) in enumerate(zip(data_predict,data_predict_confidence)):
            pre = pre.item()
            confi = confi.item()
            if pre not in predicts:
                predicts[pre] = [(i,confi)]
            else:
                predicts[pre].append((i,confi))
        for k in predicts.keys():
            predicts[k].sort(key=lambda x : x[1],reverse=True)


        topks = [1,2,3]
        for topk in topks:
            selected_index = []
            for k in predicts.keys():
                items  = [item[0] for item in predicts[k][:topk]]
                selected_index.extend(items)
            
            selected_data = dict_index(data_for_select,selected_index)
            selected_data['top1'] = selected_data['p_label'] = selected_data['label']
            
            
        


    print("first stage save model")
    torch.save(sccl_model.state_dict(),args.check_point_path)

def top_K_noisy_acc(k:float,data_len:int,noise_or_not:np.ndarray,ind_1_sorted:np.ndarray):
    top_accuracy_rm=int((1-k) * data_len)  
    top_accuracy= np.sum(noise_or_not[ind_1_sorted[top_accuracy_rm:]]) / float(data_len - top_accuracy_rm)
    return top_accuracy


def second_stage(args,sccl_model:SCCL_BERT,train_loader,train_data,data_for_select):
    moving_loss_dic = np.zeros_like(train_data['noise_or_not'], dtype=float)
    
    noisy_index = np.array([True if item else False for item in train_data['noise_or_not']],dtype=bool)
    inds_clean = []
    inds_noisy = []
    for i,(gt, p) in enumerate(zip(train_data['label'],train_data['p_label'])):
        if gt==p:
            inds_clean.append(i)
        else:
            inds_noisy.append(i)
    inds_clean = np.array(inds_clean)
    inds_noisy = np.array(inds_noisy)

    n_circles = 1
    ndata=len(train_data['text'])   
    loss_record = np.zeros(shape=[ndata,n_circles*args.n_epoch],dtype=float)  


    
    optimizer = torch.optim.Adam([
            {'params':sccl_model.sentbert.parameters()},
            {'params':sccl_model.out.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)
    sup_learner = surpervise_learner(args,sccl_model,optimizer).to(args.device)

    base_test = -1 
    mask_bools = []
    top_clean_indexes = []
    N_train = len(train_data['text'])
    for cir in range(n_circles):
        print("{}-th round".format(cir+1))
        
        for epoch in range(1,args.n_epoch+1):

            data_predict = torch.zeros(N_train,dtype=torch.long)
            data_predict_confidence = torch.zeros(N_train)

            curr_epoch = cir*args.n_epoch+(epoch-1) 
            globals_loss = 0
            sccl_model.train()
            example_loss = np.zeros_like(train_data['noise_or_not'], dtype=float)
            t = (((epoch-1) % args.n_epoch) + 1) / float(args.n_epoch) 
            lr = (1 - t) * args.max_lr + t * args.min_lr  
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr*args.lr_scale
            
            for i, batch in tqdm(enumerate(train_loader),total=len(train_loader)):
                loss,index_loss,confidences = sup_learner.forward(p_rel=batch['p_label'],text_arr=batch['text'])

                
                
                _, pred = torch.max(confidences.data, -1)  
                confidence = confidences[np.array([i for i in range(len(confidences))]),pred]
                data_predict[batch['index']] = pred.detach().cpu()
                data_predict_confidence[batch['index']] = confidence.detach().cpu()
                stop = 1
                

                
                for pi, cl in zip(batch['index'], index_loss): 
                    
                    example_loss[pi] = cl.cpu().data.item()
                    
                    loss_record[pi][curr_epoch] = cl.cpu().data.item()



                globals_loss += index_loss.sum().cpu().data.item()




            example_loss=example_loss - example_loss.mean() 
            moving_loss_dic=moving_loss_dic+example_loss

            ind_1_sorted = np.argsort(moving_loss_dic)  


            top_clean_index_epoch = []
            if args.dataset=="wiki":
                select_n = [403,2016,4032] 
            else:
                select_n = [681,3406,6812] 
            

            
            ratios = [0.01,0.02,0.025,0.03,0.04,0.05,0.06,0.07,0.075,0.08,0.09,0.1]
            if args.dataset=="wiki":
                n_train = 40320
                
            else:
                n_train = 68124
                
            select_num = [int(rt*n_train) for rt in ratios]
            accs = []
            indexes = []
            for select_n,rt in zip(select_num,ratios):
                top_clean_index = ind_1_sorted[:select_n]
                indexes.append(top_clean_index)
                p_labels = list(set((np.array(train_data['p_label'])[np.array(top_clean_index)]).tolist()))
                n_rels = len(p_labels)
                acc = cal_acc(train_data['p_label'],train_data['label'],top_clean_index)
                
                accs.append(acc)
                print("top{} {} confidence data acc= {}, n_relation:{}".format(rt,select_n,acc,n_rels))
            
            if epoch==args.n_epoch:
                ratio = [0.01,0.02,0.025,0.03,0.04,0.05,0.06,0.07,0.075,0.08,0.09,0.1]
                for index,rt in zip(indexes[:len(ratio)],ratio):
                    selected_data = dict_index(data_for_select,index)
                    acc = sum(np.array(selected_data['label'])==np.array(selected_data['p_label']))/len(selected_data['label'])
                    print("selected data acc:{}, num:{} rt:{}".format(acc,len(selected_data['text']),rt))
                    assert args.specified_save_path != ""
                    save(selected_data,os.path.join(args.specified_save_path,"O2U_{}_RT{}_SD{}.pkl".format(args.dataset,rt,args.seed)))
                    


            loss_1_sorted = moving_loss_dic[ind_1_sorted]
            remember_rate = 1 - 0.1  
            num_remember = int(remember_rate * len(loss_1_sorted))  

            noise_accuracy=np.sum(train_data['noise_or_not'][ind_1_sorted[num_remember:]]) / float(len(loss_1_sorted)-num_remember) 

            mask = np.ones_like(train_data['noise_or_not'],dtype=np.float32)  
            mask[ind_1_sorted[num_remember:]]=0  


            mask_bool = np.array(mask,dtype=bool)
            mask_bools.append(mask_bool)
            mask_selected_noisy_or_not = train_data['noise_or_not'][mask_bool]
            print("acc = ", 1-sum(mask_selected_noisy_or_not)/len(train_data['noise_or_not'])," total:",len(mask_selected_noisy_or_not))

            top_accuracy = top_K_noisy_acc(0.1,len(ind_1_sorted),train_data['noise_or_not'],ind_1_sorted)
            print ("epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / ndata, "test_accuarcy:%f" % base_test,"noise_accuracy:%f"%(noise_accuracy),"top 0.1 noise accuracy:%f"%top_accuracy)

    return mask, loss_record, noisy_index




def o2u_main(args):
    
    print(args)
    set_global_random_seed(args.seed)
    batch_size = args.batch_size
    max_len = args.max_len
    lr = args.lr
    lr_scale = args.lr_scale
    device = torch.device('cuda:{}'.format(args.cuda_index))
    args.device = device
    bert_model  = SentenceTransformer(args.model_dir)
    

    train_data,data_for_select,n_rel = prepare_neg_data(args)
    args.n_rel = n_rel

    tags = load(args.e_tags_path)




    train_dataset = Train_dataset(train_data)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    sccl_model = SCCL_BERT(bert_model,max_len,device,args.n_rel,True,tags).to(device)
    TIME = "store"
    args.check_point_path = os.path.join(CURR_DIR,"temp/temp{}.pt".format(TIME))
    first_stage(args,sccl_model=sccl_model,
                train_loader = train_loader,
                train_data = train_data,
                data_for_select = data_for_select)
    sccl_model.load_state_dict(torch.load(args.check_point_path))


    
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    args.n_epoch = 5
    mask, loss_record, noisy_index = second_stage(args,
                        sccl_model=sccl_model,
                        train_loader=train_loader,
                        train_data=train_data,
                        data_for_select = data_for_select)
    os.remove(args.check_point_path)

def prepare_neg_data(args):

    train_data1 = load(args.train_path)
    train_data1['noise_or_not'] = np.array([1 if label!=p_label else 0 for label, p_label in zip(train_data1['label'],train_data1['top1'])])
    train_data1['index'] =  [i for i in range(len(train_data1['text']))]

    
    
    train_data1['p_label'] = train_data1['top1']
    print("*"*10,"information","*"*10)
    info_acc = sum(np.array(train_data1['p_label'])==np.array(train_data1['label']))/len(train_data1['p_label'])
    n_pseudo_label_relation = len(set(train_data1['p_label']))
    print("acc:{:.4f}".format(info_acc))
    print("n_relation:{}".format(n_pseudo_label_relation))
    print("N_data:{}".format(len(train_data1['p_label'])))
    print("*"*10,"***********","*"*10)
    assert train_data1['text'][0].find("<O:")!=-1  
    


    if args.dataset == "tac": 

        train_data = {'text':train_data1['text'],
                'p_label':train_data1['top1'],
                'noise_or_not':train_data1['noise_or_not'],
                'label':train_data1['label'],
                'index':train_data1['index'],}

        return train_data, train_data1, 41  


    elif args.dataset=="wiki":
        
        train_data = {'text':train_data1['text'],
                'p_label':train_data1['top1'],
                'noise_or_not':train_data1['noise_or_not'],
                'label':train_data1['label'],
                'index':train_data1['index'],}

        return train_data, train_data1, 80

    
if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_index", type=int,default=1, help="as named")
    parser.add_argument("--batch_size", type=int,default=32, help="as named")
    
    parser.add_argument("--n_epoch", type=int,default=5, help="as named")

    parser.add_argument('--lr', type=float, default=1e-5,help='learning rate')
    parser.add_argument('--max_lr', type=float, default=5e-6,help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7,help='learning rate')
    parser.add_argument('--lr_scale', type=int, default=100, help='as named')

    parser.add_argument('--seed', type=int, default=16, help='as named')
    parser.add_argument('--model_dir', type=str, default='/data/transformers/bert-base-uncased', help='as named')
    parser.add_argument('--max_len', type=int, default=64,
                        help='length of input sentence')
    parser.add_argument("--save_info", type=str,default="", help="as named")
    parser.add_argument("--specified_save_path", type=str,default="", help="as named")


    parser.add_argument('--dataset', type=str, default="wiki", help='as named')
    parser.add_argument("--e_tags_path", type=str,default="etags.pkl", help="as named")
    parser.add_argument("--train_path", type=str,default="", help="as named")

    
    
    

    args = parser.parse_args()
    o2u_main(args)
