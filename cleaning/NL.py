
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
from utils.dict_relate import dict_index
from utils.pickle_picky import load, save
from utils.randomness import set_global_random_seed
from model.sccl import SCCL_BERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, dataset
import copy
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import argparse
import torch.utils.data as util_data
import numpy as np
from tqdm import tqdm
from collections import Counter

class Train_dataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __getitem__(self, index):
        return {
            'text': self.data['text'][index],
            'p_label': self.data['p_label'][index],  
            'index': self.data['index'][index],
        }

    def __len__(self):
        return len(self.data['text'])

def sampler(args,train_data):
    if args.dataset=='tac':
        N_training_data = 68124
    elif args.dataset=='wiki':
        N_training_data = 40320
    else:
        print('invalid dataset!')
        sys.exit()
    select_num=int(args.eta*N_training_data)
    # train_data=load(args.train_data)
    train_data['p_label'] = train_data['top1']
    label2id=load(args.label2id)
    id2label={}
    for key,value in label2id.items():
        id2label[value]=key
    p_label_confidence=train_data['confidence']
    # p_label_confidence=load(confidence_path)  # 每个label的confidence
    print(len(p_label_confidence))
    print(len(train_data['text']))
    confidence_index = np.argsort(np.array(p_label_confidence))[::-1] 

    #SELECT FIRST:  select fixed clean data based on \eta. 
    selected403  = confidence_index[:int(select_num)]
    Slabel = np.array([train_data['label'][index] for index in selected403]) # ground-truth
    Sp_label = np.array([train_data['p_label'][index] for index in selected403]) # pseudo label
    counter=Counter()
    for item in Sp_label:
        counter[item]+=1
    counter=sorted(counter.items(),key=lambda x:x[1],reverse=True)
    print('*'*50)

    n_cate = len(set(Sp_label))
    acc = sum(Slabel==Sp_label)/len(Sp_label)
    print("\ntop{} confident data acc= {}, n_classes={}, min_confidence={}".format(select_num,acc,n_cate,p_label_confidence[selected403[-1]]))


    #SELECT SECOND
    selected2=confidence_index[int(select_num):]  
    hashmap={} 
    for index in selected2:
        if train_data['p_label'][index] not in hashmap.keys():
            hashmap[train_data['p_label'][index]]=[index]
        else:
            hashmap[train_data['p_label'][index]].append(index)
    select_res=[]
    for k,v in hashmap.items():
        select_res.extend(v[:int(len(v)/len(selected2)*args.delta)])


    select_res.extend(selected403)

    Slabel = np.array([train_data['label'][index] for index in select_res]) # ground-truth
    Sp_label = np.array([train_data['p_label'][index] for index in select_res]) # pseudo label
    counter=Counter()
    for item in Sp_label:
        counter[item]+=1
    counter=sorted(counter.items(),key=lambda x:x[1],reverse=True)
    print('*'*50)
    for key,val in counter:
        print('{}_{}'.format(id2label[key],val),end='|')
    print()
    n_cate = len(set(Sp_label))
    acc = sum(Slabel==Sp_label)/len(Sp_label)
    

    
    selected_data = dict_index(train_data,select_res)
    acc = accuracy_score(selected_data['label'],selected_data['p_label'])
    print("threshold:{}, num= {}, acc= {}, n_classes:{}".format(args.delta,len(select_res),acc,n_cate))
    save(selected_data,os.path.join(args.specified_save_path,"NL_sampler_{}_delta{}_eta{}_SD{}.pkl".format(args.dataset,args.delta,args.eta,args.seed)))

def NLNL_main(args):
    print(args)
    set_global_random_seed(args.seed)
    if args.train_path.find("wiki")!=-1:
        args.dataset=mode = "wiki"
    else:
        args.dataset=mode = "tac"
    train_data = load(args.train_path)
    train_data['p_label'] = train_data['top1']


    print("*"*10,"information","*"*10)
    info_acc = sum(np.array(train_data['p_label'])==np.array(train_data['label']))/len(train_data['p_label'])
    n_pseudo_label_relation = len(set(train_data['p_label']))
    print("acc:{:.4f}".format(info_acc))
    print("n_relation:{}".format(n_pseudo_label_relation))
    print("N_data:{}".format(len(train_data['p_label'])))
    print("*"*10,"***********","*"*10)
    

    num_classes = args.n_rel
    args.N_train = N_train = len(train_data['text'])
    train_data['index'] = [i for i in range(args.N_train)]
    train_dataset = Train_dataset(train_data)
    train_loader = util_data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    inds_noisy = np.asarray([index for index in range(len(train_data['p_label'])) if train_data['p_label'][index]!=train_data['label'][index]  ])
    inds_clean = np.delete(np.arange(N_train), inds_noisy)
    tags = load(args.e_tags_path)
    device = torch.device('cuda:{}'.format(args.cuda_index))
    args.device = device
    bert_model = SentenceTransformer(args.model_dir)
    sccl_model:SCCL_BERT = SCCL_BERT(bert_model, args.max_len,
                           device, args.n_rel, True, tags).to(device)

    
    
    
    optimizer = torch.optim.AdamW([
        {'params': sccl_model.sentbert.parameters()},
        {'params': sccl_model.out.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)
    

    
    weight = torch.FloatTensor(num_classes).zero_() + 1.
    for i in range(num_classes):
        weight[i] = (torch.from_numpy(np.array(train_data['p_label']).astype(int)) == i).sum()  
    weight = 1 / (weight / weight.max()) 



    

    criterion = nn.CrossEntropyLoss()
    criterion_nll = nn.NLLLoss()
    criterion_nr = nn.CrossEntropyLoss(reduction='none')  
    criterion.to(device)
    criterion_nll.to(device)
    criterion_nr.to(device)
    

    
    
    train_preds = torch.zeros(N_train, num_classes) - 1.
    num_hist = 10
    train_preds_hist = torch.zeros(N_train, num_hist, num_classes)   
    pl_ratio = 0.
    nl_ratio = 1.-pl_ratio  
    train_losses = torch.zeros(N_train) - 1.  
    

    
    
    for epoch in range(args.epoch):
        train_loss = train_loss_neg = train_acc = 0.0
        pl = 0.; nl = 0.
        sccl_model.train()
        accs = []
        losses = []
        data_predict = torch.zeros(N_train,dtype=torch.long) 
        data_predict_confidence = torch.zeros(N_train) 
        for i, data in enumerate(train_loader):
            
            text, labels, index = data['text'],data['p_label'],data['index']
            labels_neg = (labels.unsqueeze(-1).repeat(1, args.ln_neg)
                      + torch.LongTensor(len(labels), args.ln_neg).random_(1, num_classes)) % num_classes
            assert labels_neg.max() <= num_classes-1
            assert labels_neg.min() >= 0
            assert (labels_neg != labels.unsqueeze(-1).repeat(1, args.ln_neg)
                    ).sum() == len(labels)*args.ln_neg  
            labels = labels.to(device)
            labels_neg = labels_neg.to(device)
            logits = sccl_model.out(sccl_model.get_embeddings_PURE(text))

            s_neg = torch.log(torch.clamp(1.-F.softmax(logits, -1), min=1e-5, max=1.))  
            s_neg *= weight[labels].unsqueeze(-1).expand(s_neg.size()).to(device)
            _, pred = torch.max(logits.data, -1)  

            
            
            confidences = F.softmax(logits,-1)
            confidence = confidences[np.array([i for i in range(len(logits))]),pred]
            data_predict[index] = pred.detach().cpu() 
            data_predict_confidence[index] = confidence.detach().cpu() 
            stop = 1
            

            acc = float((pred == labels.data).sum())   
            train_acc += acc
            accs.append(acc/len(index))

            train_loss += logits.size(0)*criterion(logits, labels).data
            train_loss_neg += logits.size(0) * criterion_nll(s_neg, labels_neg[:, 0]).data

            train_losses[index] = criterion_nr(logits, labels).cpu().data  
            

            labels = labels*0 - 100  
            
            loss_neg = criterion_nll(s_neg.repeat(args.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
            # loss_pl = criterion(logits, labels)* float((labels >= 0).sum())
            
            loss = (loss_neg) / (float((labels >= 0).sum()) +float((labels_neg[:, 0] >= 0).sum()))
            loss.backward()
            optimizer.step()
            l = logits.size(0)*loss.detach().cpu().data
            train_loss+=l
            
            losses.append(l/logits.size(0))
            train_preds[index.cpu()] = F.softmax(logits, -1).cpu().data

            pl += float((labels >= 0).sum())
            
            print("EPOCH[{}] step {}/{} ,  loss: {:.4f} train_acc: {:.4f}  ".format(epoch+1,i+1,len(train_loader),np.mean(losses),np.mean(accs)))
            nl += float((labels_neg[:, 0] >= 0).sum())
            


        
        
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
            selected_label = [train_data['label'][i] for i in selected_index]
            
            selected_data = dict_index(train_data,selected_index)
            selected_data['top1'] = selected_data['p_label'] = selected_data['label'] 
            
            
        



        train_loss /= N_train
        train_loss_neg /= N_train
        train_acc /= N_train
        pl_ratio = pl / float(N_train)

        
        assert train_preds[train_preds < 0].nelement() == 0
        train_preds_hist[:, epoch % num_hist] = train_preds
        train_preds = train_preds*0 - 1.
        assert train_losses[train_losses < 0].nelement() == 0
        train_losses = train_losses*0 - 1.
        

        
        
        
        p_label_confidence = train_preds_hist.mean(1)[torch.arange(N_train), np.array(train_data['p_label']).astype(int)] 
        confidence_index = np.argsort(np.array(p_label_confidence))[::-1]  
        indexes = []
        ratios = [0.01,0.02,0.025,0.03,0.04,0.05,0.06,0.07,0.075,0.08,0.09,0.1]
        if mode=="wiki":
            n_train = 40320
        else:
            n_train = 68124
        select_num = [int(rt*n_train) for rt in ratios]
        accs = []
        for select_n,rt in zip(select_num,ratios):
            selected403  = confidence_index[:int(select_n)]
            indexes.append(selected403)
            
            Slabel = np.array([train_data['label'][index] for index in selected403]) 
            Sp_label = np.array([train_data['p_label'][index] for index in selected403]) 
            n_cate = len(set(Sp_label))
            acc = sum(Slabel==Sp_label)/len(Sp_label)
            accs.append(acc)
            print("top confident {} data, num={}, acc= {}, n_relation:{}".format(rt,select_n,acc,n_cate))
        if epoch+1==args.epoch:
            
            ratio = [0.01,0.02,0.025,0.03,0.04,0.05,0.06,0.07,0.075,0.08,0.09,0.1]
            for index,rt in zip(indexes[:len(ratio)],ratio):
                selected_data = dict_index(train_data,index)
                acc = sum(np.array(selected_data['label'])==np.array(selected_data['top1']))/len(selected_data['label'])
                print("selected data acc:{}, num:{}".format(acc,len(selected_data['text'])))
                assert args.specified_save_path != ""
                save(selected_data,os.path.join(args.specified_save_path,"NL_{}_RT{}_SD{}.pkl".format(args.dataset,rt,args.seed)))
            train_data['confidence'] = np.array(p_label_confidence)
            sampler(args,train_data)
            

        



        
        if args.plot:
            clean_plot = train_preds_hist.mean(1)[torch.arange(N_train)[
                inds_clean], np.array(train_data['p_label']).astype(int)[inds_clean]]
            noisy_plot = train_preds_hist.mean(1)[torch.arange(N_train)[
                inds_noisy], np.array(train_data['p_label']).astype(int)[inds_noisy]]
            clean_plot = clean_plot.numpy()
            noisy_plot = noisy_plot.numpy()
            plt.hist(clean_plot, bins=33, edgecolor='black', alpha=0.5,
                    range=(0, 1), label='clean', histtype='bar')
            plt.hist(noisy_plot, bins=33, edgecolor='black', alpha=0.5,
                    range=(0, 1), label='noisy', histtype='bar')
            plt.xlabel('probability')
            plt.ylabel('number of data')
            plt.grid()
            plt.legend()
            img_dir = os.path.join(CURR_DIR,'output_imgs/{}_confidence_distribution_epoch{}_T{}.jpg'.format(mode,epoch,TIME))
            print("img here: ",img_dir)
            plt.savefig(img_dir)
            plt.clf()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=16, help="as named")
    parser.add_argument("--cuda_index", type=int, default=3, help="as named")
    parser.add_argument('--lr', type=float, default=4e-7, help='learning rate')
    parser.add_argument('--lr_scale', type=int, default=100, help='as named')
    parser.add_argument('--epoch', type=int, default=10, help='as named')
    parser.add_argument("--specified_save_path", type=str,default="", help="selected clean data will be stored here")


    parser.add_argument('--n_rel', type=int, default=41, help='as named') 
    parser.add_argument("--train_path", type=str,
                        default="tac_annotation.pkl", help="as named")
    parser.add_argument("--e_tags_path", type=str,
                        default="tags.pkl", help="as named")
    parser.add_argument("--save_dir", type=str,
                        default="outputs", help="as named")
    parser.add_argument('--ln_neg', type=int, default=41,
                        help='number of negative labels on single image for training (ex. 110 for cifar100)')
    # sampler argument 
    parser.add_argument("--eta", type=float, help="as named")
    parser.add_argument("--delta", type=int, help="as named")
    parser.add_argument("--dataset", type=str, default=None, help="as named")
    parser.add_argument("--label2id", type=str, default=None, help="as named")

    parser.add_argument('--save_info', type=str,
                        default="", help='as named')
    parser.add_argument('--model_dir', type=str,
                        default='/data/transformers/bert-base-uncased', help='as named')
    parser.add_argument('--max_len', type=int, default=64,
                        help='length of input sentence')
    parser.add_argument('--plot', type=bool, default=True)
    args = parser.parse_args()



    NLNL_main(args)

    
