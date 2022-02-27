import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())

sys.path.append(CURR_DIR)
P = PATH.parent
print("current dir: ",CURR_DIR)
for i in range(1):  # add parent path, height = 3
    P = P.parent
    PROJECT_PATH = str(P.absolute())
    sys.path.append(str(P.absolute()))
import time
TIME=time.strftime("%m%d%H%M%S", time.localtime())# 记录被初始化的时间
print("time",TIME)

from torch.utils.data import Dataset, dataset
import torch.utils.data as util_data
from tqdm import tqdm
import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer
from mymodel.sccl import SCCL_BERT
from utils.dict_relate import dict_index
from utils.randomness import set_global_random_seed
from clean_data.clean import get_format_train_text, find_uppercase
from utils.pickle_picky import load,save
from utils.metric import cal_acc
from  torch.utils import data
import torch.nn.functional as F
import copy
import random
import numpy as np
import argparse
from tqdm import tqdm



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

class mydataset(Dataset):  # dev 使用
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
        loss = self.CEloss(out,p_rel)  # 在这里加上你的loss
        index_loss = loss.clone()
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        index_loss = index_loss.detach()
        return loss.detach(),index_loss



"""
代码中几个比较重要的点:
    1.example_loss: 在某个epoch中, 每个数据的loss , 长度和整个数据集的长度一样
Second_stage:
    2,moving_loss: 在整个stage中, 每个epoch的example_loss 都叠加进来(example_loss先要经过减去mean的操作)

"""

def adjust_learning_rate(optimizer,epoch,max_epoch,scale):
    if epoch<0.25*max_epoch:
        lr = (5e-6)*3
    elif epoch < 0.5 * max_epoch:
        lr = (5e-6)*2
    else:
        lr = 5e-6
    # if epoch<0.25*max_epoch:
    #     lr = (2e-5)*10
    # elif epoch < 0.5 * max_epoch:
    #     lr = (2e-5)*5
    # else:
        # lr = 2e-5
    # lr = 5e-5
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr*scale
    return lr
def first_stage(args,sccl_model:SCCL_BERT,train_loader,train_data):
    """
        first_stage 负责先训练模型
        
    """
    ndata=len(train_data['text'])  # 注意这里要改
    optimizer = torch.optim.Adam([
            {'params':sccl_model.sentbert.parameters()},
            {'params':sccl_model.out.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)
    sup_learner = surpervise_learner(args,sccl_model,optimizer).to(args.device)
    base_test = -1 # acc


    print()
    for epoch in range(1, args.n_epoch+1): # args.n_epoch
        globals_loss = 0
        sccl_model.train()
        lr=adjust_learning_rate(optimizer,epoch,args.n_epoch,args.lr_scale) 
        for i, batch in tqdm(enumerate(train_loader),total=len(train_loader)):
            loss,index_loss = sup_learner.forward(p_rel=batch['p_label'],text_arr=batch['text'])
            globals_loss += index_loss.sum().cpu().data.item()
        print ("epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss /ndata, "test_accuarcy:%f" % base_test)
    print("first stage save model")
    torch.save(sccl_model.state_dict(),args.check_point_path)

def top_K_noisy_acc(k:float,data_len:int,noise_or_not:np.ndarray,ind_1_sorted:np.ndarray):
    top_accuracy_rm=int((1-k) * data_len)  # 40000
    top_accuracy= np.sum(noise_or_not[ind_1_sorted[top_accuracy_rm:]]) / float(data_len - top_accuracy_rm)
    return top_accuracy


def second_stage(args,sccl_model:SCCL_BERT,train_loader,train_data,data_for_select):
    moving_loss_dic = np.zeros_like(train_data['noise_or_not'], dtype=float)
    # 找出noisy index 以及非noisy index
    noisy_index = np.array([True if item else False for item in train_data['noise_or_not']],dtype=bool)

    n_circles = 1
    ndata=len(train_data['text'])   # 注意这里要改
    loss_record = np.zeros(shape=[ndata,n_circles*args.n_epoch],dtype=float)  #用于记录每个数据在


    
    optimizer = torch.optim.Adam([
            {'params':sccl_model.sentbert.parameters()},
            {'params':sccl_model.out.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)
    sup_learner = surpervise_learner(args,sccl_model,optimizer).to(args.device)

    base_test = -1 # acc
    mask_bools = []
    top_clean_indexes = []
    for cir in range(n_circles):
        print("{}-th round".format(cir+1))
        # moving_loss_dic = np.zeros_like(train_data['noise_or_not'], dtype=float)  # 尝试every circle 清零
        for epoch in range(1,args.n_epoch+1):
            curr_epoch = cir*args.n_epoch+(epoch-1) # start from zero 
            globals_loss = 0
            sccl_model.train()
            example_loss = np.zeros_like(train_data['noise_or_not'], dtype=float)
            t = (((epoch-1) % args.n_epoch) + 1) / float(args.n_epoch) #  10? 10 is the total number of epoch of a cycilcal round
            lr = (1 - t) * args.max_lr + t * args.min_lr  # 0.01 is max lr and 0.001 is the min one
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr*args.lr_scale
            # train it right now
            for i, batch in tqdm(enumerate(train_loader),total=len(train_loader)):
                loss,index_loss = sup_learner.forward(p_rel=batch['p_label'],text_arr=batch['text'])

                # 记录example loss
                for pi, cl in zip(batch['index'], index_loss): # indexes?
                    # 遍历loss和其index
                    example_loss[pi] = cl.cpu().data.item()
                    # 记录loss全纪录
                    loss_record[pi][curr_epoch] = cl.cpu().data.item()



                globals_loss += index_loss.sum().cpu().data.item()

            example_loss=example_loss - example_loss.mean() #? 这什么操作?  某种程度上的归一化操作
            moving_loss_dic=moving_loss_dic+example_loss

            ind_1_sorted = np.argsort(moving_loss_dic)  # 给这个全局的loss sort一波(从小到大, index)

            # 取出loss最小的数据
            # top_clean_index_epoch = []
            # for topk_less in [0.001,0.005,0.01,0.02,0.05,0.10,0.15,0.20,0.3,0.4,0.5]:
            #     top_clean_n = int(len(ind_1_sorted)*topk_less)
            #     top_clean_index = ind_1_sorted[:top_clean_n] # 选出了topk_less小的数据的index
            #     p_labels = list(set((np.array(train_data['p_label'])[np.array(top_clean_index)]).tolist()))
            #     n_rels = len(p_labels)
            #     sorted(p_labels)
            #     top_clean_index_epoch.append(top_clean_index) # 保存当前epoch的topk index
            #     acc = cal_acc(train_data['p_label'],train_data['label'],top_clean_index)
            #     save(top_clean_index,"/home/tywang/myURE/URE/O2U_bert/out/top_clean_index_train_tac_num{}_k{}_acc{:.4f}.pkl".format(top_clean_n,topk_less,acc))
            top_clean_index_epoch = []
            if args.dataset=="wiki":
                select_n = [403,2016,4032] # number corresponding to(0.01, 0.05, 0.1, 1.0)*n_train (n_train=40320)
            else:
                select_n = [681,3406,6812] # number corresponding to(0.01, 0.05, 0.1, 1.0)*n_train (n_train=68124)
            # assert args.dataset=="wiki"
            for top_clean_n,rt in zip(select_n,[0.01,0.05,0.1]):
                top_clean_index = ind_1_sorted[:top_clean_n] # 选出了topk_less小的数据的index
                # 计算选出的数据的类别数(往往不全
                p_labels = list(set((np.array(train_data['p_label'])[np.array(top_clean_index)]).tolist()))
                n_rels = len(p_labels)
                # 给p_label排序, 方便打印出易于辨别的类别变化
                sorted(p_labels)
                top_clean_index_epoch.append(top_clean_index) # 保存当前epoch的topk index
                acc = cal_acc(train_data['p_label'],train_data['label'],top_clean_index)
                if epoch==args.n_epoch:
                    selected_data = dict_index(data_for_select,top_clean_index)
                    ##
                    # basic information of selected data
                    acc = sum(np.array(selected_data['label'])==np.array(selected_data['top1']))/len(selected_data['label'])
                    print("selected data acc:{}".format(acc))
                    ##
                    save(selected_data,os.path.join(PROJECT_PATH,"finetune/selected_data/{}/selected_n{}train_O2U.pkl".format(args.dataset,rt))) 
                print("top ,number:{} acc:{:.4f}  n_rel:{}".format(top_clean_n,acc,n_rels)," ",p_labels)
            
            # top_clean_indexes.append(top_clean_index_epoch)
            # save(top_clean_indexes,"/home/tywang/myURE/URE/O2U_bert/out/top_clean_indexes.pkl")

            loss_1_sorted = moving_loss_dic[ind_1_sorted]
            remember_rate = 1 - 0.1  # 0.8  
            num_remember = int(remember_rate * len(loss_1_sorted))  # 选出筛选出来的number

            # 看ground truth检测选出来的 0.1 当中是noisy的比例
            # 看ground truth 的前0.1 loss大的数据
            # noise_or_not 当中, 1 表示是noisy
            noise_accuracy=np.sum(train_data['noise_or_not'][ind_1_sorted[num_remember:]]) / float(len(loss_1_sorted)-num_remember) 

            mask = np.ones_like(train_data['noise_or_not'],dtype=np.float32)  # mask=1 => selected
            mask[ind_1_sorted[num_remember:]]=0  # 后面的这些就归为noisy数据   mask = 0 => dropped

            # 计算使用了MASK之后的准确率
            mask_bool = np.array(mask,dtype=bool)
            mask_bools.append(mask_bool)
            mask_selected_noisy_or_not = train_data['noise_or_not'][mask_bool]
            print("acc = ", 1-sum(mask_selected_noisy_or_not)/len(train_data['noise_or_not'])," total:",len(mask_selected_noisy_or_not))

            # 再检查 排名top k% 的数据的noisy的准确率
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
    # data_loader

    train_data,data_for_select,n_rel = prepare_neg_data(args)
    args.n_rel = n_rel

    tags = load(args.e_tags_path)




    train_dataset = Train_dataset(train_data)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    sccl_model = SCCL_BERT(bert_model,max_len,device,args.n_rel,True,tags).to(device)
    args.check_point_path = os.path.join(CURR_DIR,"temp/temp.pt")
    first_stage(args,sccl_model=sccl_model,
                train_loader = train_loader,
                train_data = train_data)
    sccl_model.load_state_dict(torch.load(args.check_point_path))


    # 按照更小的batch_size重新生成train_loader
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    args.n_epoch = 5
    mask, loss_record, noisy_index = second_stage(args,
                        sccl_model=sccl_model,
                        train_loader=train_loader,
                        train_data=train_data,
                        data_for_select = data_for_select)

def prepare_neg_data(args):
    """
    准备neg数据
        输入的数据的要求:
        train: key: dict_keys(['text', 'rel', 'subj', 'obj', 'subj_type', 'obj_type',
                             'top1', 'top2', 'label', 'pos_or_not', 'noise_or_not'])
        test: key:  dict_keys(['text', 'rel', 'subj', 'obj', 'subj_type', 'obj_type', 
                              'label', 'pos_or_not'])
        text 必须是以及经过标准化的
    返回: 符合要求的train, test, 以及n_rel
    """

    """
    train_demo:
    'text':['<O:PERSON> Tom Thaba...election .', 'In 1983 , a year aft...undation .', 'This was among a bat...IZATION> .', 'The latest investiga...pipeline .', 'The event is a respo...IZATION> .', 'Manning was prime mi...IZATION> .', '<O:PERSON> Christine...<S:PERSON>', 'Al-Hubayshi explaine...passport .', '<S:PERSON> Olivia Pa...ay there !', 'But US and Indian ex...al India .', "`` I have n... website .', "In O'Brien ...g others .', 'Folded into <S:PERSO...unseling .', 'So let me get this s... in 2009 ?', ...]
    'rel':['org:founded_by', 'no_relation', 'no_relation', 'no_relation', 'no_relation', 'no_relation', 'no_relation', 'no_relation', 'per:employee_of', 'org:alternate_names', 'no_relation', 'no_relation', 'no_relation', 'no_relation', ...]
    'subj':['All Basotho Convention', 'Forsberg', 'OUP', 'Fyffes', 'New York Immigration Coalition', 'UNC', 'Haddad-Adel', 'he', 'Olivia Palermo', 'Lashkar-e-Taiba', 'Castro', 'Jerome Robbins', 'Dillinger', 'he', ...]
    'obj':['Tom Thabane', 'John D.', 'Oxford World', '106 million', 'March', 'National Alliance fo...nstruction', 'Christine Egerszegi-Obrist', 'he', 'Gossip Girl', 'Army of the Pure', 'Iranian', 'Broadway', 'director', '2001', ...]
    'subj_type':['ORGANIZATION', 'PERSON', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'PERSON', 'PERSON', 'PERSON', 'ORGANIZATION', 'PERSON', 'PERSON', 'PERSON', 'PERSON', ...]
    'obj_type':['PERSON', 'PERSON', 'ORGANIZATION', 'NUMBER', 'DATE', 'ORGANIZATION', 'PERSON', 'PERSON', 'ORGANIZATION', 'ORGANIZATION', 'NATIONALITY', 'LOCATION', 'TITLE', 'DATE', ...]
    'top1':[37, 41, 41, 41, 41, 6, 41, 41, 41, 19, 41, 41, 41, 41, ...]
    'top2':[6, 23, 19, 6, 7, 14, 16, 27, 15, 6, 27, 23, 23, 27, ...]
    'label':[37, 41, 41, 41, 41, 41, 41, 41, 8, 19, 41, 41, 41, 41, ...]
    'pos_or_not':[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, ...]
    'index':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...]
    
    """
    train_data1 = load(args.train_path)
    train_data1['noise_or_not'] = np.array([1 if label!=p_label else 0 for label, p_label in zip(train_data1['label'],train_data1['top1'])])
    train_data1['index'] =  [i for i in range(len(train_data1['text']))]

    ##
    # print basic information
    train_data1['p_label'] = train_data1['top1']
    print("*"*10,"information","*"*10)
    info_acc = sum(np.array(train_data1['p_label'])==np.array(train_data1['label']))/len(train_data1['p_label'])
    n_pseudo_label_relation = len(set(train_data1['p_label']))
    print("acc:{:.4f}".format(info_acc))
    print("n_relation:{}".format(n_pseudo_label_relation))
    print("N_data:{}".format(len(train_data1['p_label'])))
    print("*"*10,"***********","*"*10)
    assert train_data1['text'][0].find("<O:")!=-1  # make sure data is tagged with <O:OBJECT_TYPE>
    ##


    if args.dataset == "tac": # 仅仅pos的类

        train_data = {'text':train_data1['text'],
                'p_label':train_data1['top1'],
                'noise_or_not':train_data1['noise_or_not'],
                'label':train_data1['label'],
                'index':train_data1['index'],}

        return train_data, train_data1, 41  # data for train, data for selection, n_relation


    elif args.dataset=="wiki":
        
        train_data = {'text':train_data1['text'],
                'p_label':train_data1['top1'],
                'noise_or_not':train_data1['noise_or_not'],
                'label':train_data1['label'],
                'index':train_data1['index'],}

        return train_data, train_data1, 80

    
if __name__=="__main__":
    """
    train_demo:
    'text':['<O:PERSON> Tom Thaba...election .', 'In 1983 , a year aft...undation .', 'This was among a bat...IZATION> .', 'The latest investiga...pipeline .', 'The event is a respo...IZATION> .', 'Manning was prime mi...IZATION> .', '<O:PERSON> Christine...<S:PERSON>', 'Al-Hubayshi explaine...passport .', '<S:PERSON> Olivia Pa...ay there !', 'But US and Indian ex...al India .', "`` I have n... website .', "In O'Brien ...g others .', 'Folded into <S:PERSO...unseling .', 'So let me get this s... in 2009 ?', ...]
    'rel':['org:founded_by', 'no_relation', 'no_relation', 'no_relation', 'no_relation', 'no_relation', 'no_relation', 'no_relation', 'per:employee_of', 'org:alternate_names', 'no_relation', 'no_relation', 'no_relation', 'no_relation', ...]
    'subj':['All Basotho Convention', 'Forsberg', 'OUP', 'Fyffes', 'New York Immigration Coalition', 'UNC', 'Haddad-Adel', 'he', 'Olivia Palermo', 'Lashkar-e-Taiba', 'Castro', 'Jerome Robbins', 'Dillinger', 'he', ...]
    'obj':['Tom Thabane', 'John D.', 'Oxford World', '106 million', 'March', 'National Alliance fo...nstruction', 'Christine Egerszegi-Obrist', 'he', 'Gossip Girl', 'Army of the Pure', 'Iranian', 'Broadway', 'director', '2001', ...]
    'subj_type':['ORGANIZATION', 'PERSON', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'PERSON', 'PERSON', 'PERSON', 'ORGANIZATION', 'PERSON', 'PERSON', 'PERSON', 'PERSON', ...]
    'obj_type':['PERSON', 'PERSON', 'ORGANIZATION', 'NUMBER', 'DATE', 'ORGANIZATION', 'PERSON', 'PERSON', 'ORGANIZATION', 'ORGANIZATION', 'NATIONALITY', 'LOCATION', 'TITLE', 'DATE', ...]
    'top1':[37, 41, 41, 41, 41, 6, 41, 41, 41, 19, 41, 41, 41, 41, ...]
    'top2':[6, 23, 19, 6, 7, 14, 16, 27, 15, 6, 27, 23, 23, 27, ...]
    'label':[37, 41, 41, 41, 41, 41, 41, 41, 8, 19, 41, 41, 41, 41, ...]
    'index':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...]
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_index", type=int,default=1, help="as named")
    parser.add_argument("--batch_size", type=int,default=32, help="as named")
    # 是否为2分类的O2U
    parser.add_argument("--n_epoch", type=int,default=5, help="as named")

    parser.add_argument('--lr', type=float, default=1e-5,help='learning rate')
    parser.add_argument('--max_lr', type=float, default=5e-6,help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7,help='learning rate')
    parser.add_argument('--lr_scale', type=int, default=100, help='as named')

    parser.add_argument('--seed', type=int, default=16, help='as named')
    parser.add_argument('--model_dir', type=str, default='/data/transformers/bert-base-uncased', help='as named')
    parser.add_argument('--max_len', type=int, default=64,
                        help='length of input sentence')



    """wiki"""
    # parser.add_argument('--dataset', type=str, default="wiki", help='as named')
    # parser.add_argument("--e_tags_path", type=str,default="/home/tywang/myURE/URE/WIKI/typed/etags.pkl", help="as named")
    # parser.add_argument("--train_path", type=str,default="/home/tywang/myURE/URE_mnli/temp_files/analysis_0.01510/wiki_FewShot_train_num40320_top1_0.4468_Xlarge.pkl", help="as named")
    """tac"""
    parser.add_argument('--dataset', type=str, default="tac", help='as named')
    parser.add_argument("--e_tags_path", type=str,default="/home/tywang/URE-master/data/tac/tags.pkl", help="as named")
    parser.add_argument("--train_path", type=str,default="/home/tywang/URE-master/data/tac/annotated/train_num9710_top1_0.5799.pkl", help="as named")

    args = parser.parse_args()
    o2u_main(args)
