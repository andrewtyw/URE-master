
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
TIME=time.strftime("%m%d%H%M%S", time.localtime())# record the initial time
print("time",TIME)
from utils.dict_relate import dict_index,dict2list
from utils.pickle_picky import load, save
from utils.randomness import set_global_random_seed
from model.sccl import SCCL_BERT
from sentence_transformers import SentenceTransformer
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
            'p_label': self.data['p_label'][index],  # pseudo_label
            'index': self.data['index'][index],
        }

    def __len__(self):
        return len(self.data['text'])




def NLNL_main(args):
    print(args)
    set_global_random_seed(args.seed)
    if args.train_path.find("tac")!=-1:
        args.dataset=mode = "tac"
    else:
        args.dataset=mode = "wiki"
    train_data = load(args.train_path)
    train_data['p_label'] = train_data['top1']
    ##
    # print basic information of the train_data
    print("*"*10,"information","*"*10)
    info_acc = sum(np.array(train_data['p_label'])==np.array(train_data['label']))/len(train_data['p_label'])
    n_pseudo_label_relation = len(set(train_data['p_label']))
    print("acc:{:.4f}".format(info_acc))
    print("n_relation:{}".format(n_pseudo_label_relation))
    print("N_data:{}".format(len(train_data['p_label'])))
    print("*"*10,"***********","*"*10)
    ##

    num_classes = args.n_rel
    args.N_train = N_train = len(train_data['text'])
    train_data['index'] = [i for i in range(args.N_train)]
    train_data_list = dict2list(train_data)
    train_dataset = Train_dataset(train_data)
    # test_dataset = Test_dataset(test_data)
    train_loader = util_data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    inds_noisy = np.asarray([index for index in range(len(train_data['p_label'])) if train_data['p_label'][index]!=train_data['label'][index]  ])
    inds_clean = np.delete(np.arange(N_train), inds_noisy)
    tags = load(args.e_tags_path)
    device = torch.device('cuda:{}'.format(args.cuda_index))
    args.device = device
    bert_model = SentenceTransformer(args.model_dir)
    sccl_model:SCCL_BERT = SCCL_BERT(bert_model, args.max_len,
                           device, args.n_rel, True, tags).to(device)

    ##
    # set training related
    optimizer = torch.optim.AdamW([
        {'params': sccl_model.sentbert.parameters()},
        {'params': sccl_model.out.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)
    ##

    # 产生weight
    weight = torch.FloatTensor(num_classes).zero_() + 1.
    for i in range(num_classes):
        weight[i] = (torch.from_numpy(np.array(train_data['p_label']).astype(int)) == i).sum()  
    weight = 1 / (weight / weight.max())  



    ##
    # criterions
    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = nn.CrossEntropyLoss()
    criterion_nll = nn.NLLLoss()
    criterion_nr = nn.CrossEntropyLoss(reduction='none')  # compute per-sample losses
    criterion.to(device)
    criterion_nll.to(device)
    criterion_nr.to(device)
    ##
    n_print_steps = 10
    ##
    # NLNL parameters
    train_preds = torch.zeros(N_train, num_classes) - 1.
    num_hist = 10
    train_preds_hist = torch.zeros(N_train, num_hist, num_classes)  
    train_preds_hist_all = torch.zeros( args.epoch, N_train,num_classes) 
    train_preds_hist_all_list = []
    pl_ratio = 0.
    nl_ratio = 1.-pl_ratio 
    train_losses = torch.zeros(N_train) - 1. 
    ##
    ##
    # train
    best_test_acc = 0.0
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

            ##
            # find labels for fewshot
            confidences = F.softmax(logits,-1)
            confidence = confidences[np.array([i for i in range(len(logits))]),pred]
            data_predict[index] = pred.detach().cpu() 
            data_predict_confidence[index] = confidence.detach().cpu() 
            stop = 1
            ##

            acc = float((pred == labels.data).sum())   
            train_acc += acc
            accs.append(acc/len(index))

            train_loss += logits.size(0)*criterion(logits, labels).data
            train_loss_neg += logits.size(0) * criterion_nll(s_neg, labels_neg[:, 0]).data

            train_losses[index] = criterion_nr(logits, labels).cpu().data
            

            labels = labels*0 - 100  # In the program, we do not use the process of SelNL and SelPL, cause they will make lower accuracy in "clean data"
            
            loss_neg = criterion_nll(s_neg.repeat(args.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
            loss_pl = criterion(logits, labels)* float((labels >= 0).sum())
            
            loss = (loss_neg) / (float((labels >= 0).sum()) +float((labels_neg[:, 0] >= 0).sum()))
            loss.backward()
            optimizer.step()
            l = logits.size(0)*loss.detach().cpu().data
            train_loss+=l
            optimizer.zero_grad()
            losses.append(l/logits.size(0))
            train_preds[index.cpu()] = F.softmax(logits, -1).cpu().data

            pl += float((labels >= 0).sum())

            if i%n_print_steps==0:
                print(" step {}/{} ,  loss_{:.4f} acc_{:.4f}  ".format(i+1,len(train_loader),np.mean(losses),np.mean(accs)))
            nl += float((labels_neg[:, 0] >= 0).sum())
            # if i==10:break


        ## it is used to select data for fewshot.
        # select topk confident data of each category in predition 
        predicts = dict()
        for i,(pre,confi) in enumerate(zip(data_predict,data_predict_confidence)):
            pre = pre.item()
            confi = confi.item()
            if pre not in predicts:
                predicts[pre] = [(i,confi)] # (data_index, confidence)
            else:
                predicts[pre].append((i,confi))
        for k in predicts.keys():  # sort it 
            predicts[k].sort(key=lambda x : x[1],reverse=True)


        topks = [1,2,3]
        for topk in topks:
            selected_index = []
            for k in predicts.keys():
                items  = [item[0] for item in predicts[k][:topk]]
                selected_index.extend(items) 
            selected_label = [train_data['label'][i] for i in selected_index]
            # print(Counter(selected_label))
            selected_data = dict_index(train_data,selected_index)
            selected_data['top1'] = selected_data['p_label'] = selected_data['label'] 
            # print(Counter(selected_data['top1']))
        ##



        train_loss /= N_train
        train_loss_neg /= N_train
        train_acc /= N_train
        pl_ratio = pl / float(N_train)
        nl_ratio = nl / float(N_train)
        noise_ratio = 1. - pl_ratio

        noise = (np.array(train_data['p_label']).astype(int) != np.array(train_data['label'])).sum()
        print()
        print('[%6d/%6d] loss: %5f, loss_neg: %5f, acc: %5f, lr: %5f, noise: %d, pl: %5f, nl: %5f, noise_ratio: %5f'
                % (epoch, args.epoch, train_loss, train_loss_neg, np.mean(accs), args.lr, noise, pl_ratio, nl_ratio, noise_ratio))



        ###############################################################################################
        assert train_preds[train_preds < 0].nelement() == 0
        train_preds_hist[:, epoch % num_hist] = train_preds
        train_preds_hist_all[epoch] = train_preds
        train_preds = train_preds*0 - 1.
        assert train_losses[train_losses < 0].nelement() == 0
        train_losses = train_losses*0 - 1.
        ###############################################################################################


        
        ##  
        p_label_confidence = train_preds_hist.mean(1)[torch.arange(N_train), np.array(train_data['p_label']).astype(int)] # shape = N_train
        train_preds_hist_all_list.append(p_label_confidence.numpy())
        metric = {
            "his_all":train_preds_hist_all,
            "his_all_epo":train_preds_hist_all_list,
            "his":train_preds_hist,
            "noisy_metric":p_label_confidence,
            "train_data":train_data,
            "train_data_list":train_data_list
        }
        save(metric,args.specified_save_path)

        if True:
            g_data2 = train_preds_hist.mean(1)[torch.arange(N_train), np.array(train_data['p_label'])]
            g_data2 = g_data2.numpy()
            plt.hist(g_data2, bins=33, edgecolor='black', alpha=0.5, label='clean', histtype='bar')
            plt.xlabel('probability')
            plt.ylabel('number of data')
            plt.grid()
            plt.legend()
            img_dir = os.path.join(CURR_DIR,'imgs/{}_wikifact_his{}_T{}.jpg'.format(mode,epoch,TIME))
            print("img is here: ",img_dir)
            plt.savefig(img_dir)
            plt.clf()






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=16, help="as named")
    parser.add_argument("--cuda_index", type=int, default=3, help="as named")
    parser.add_argument('--lr', type=float, default=4e-7, help='learning rate')
    parser.add_argument('--lr_scale', type=int, default=100, help='as named')
    parser.add_argument('--epoch', type=int, default=100, help='as named')
    parser.add_argument("--specified_save_path", type=str,default="", help="as named")

    parser.add_argument('--n_rel', type=int, default=41, help='as named')
    parser.add_argument("--train_path", type=str,
                        default="", help="as named")
    parser.add_argument("--e_tags_path", type=str,
                        default="", help="as named")
    parser.add_argument('--ln_neg', type=int, default=41,
                        help='')


    """communal"""

    parser.add_argument('--save_info', type=str,
                        default="", help='as named')
    parser.add_argument('--model_dir', type=str,
                        default='', help='as named')
    parser.add_argument('--max_len', type=int, default=64,
                        help='length of input sentence')
    args = parser.parse_args()



    NLNL_main(args)

    