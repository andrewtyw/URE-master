
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
from utils.dict_relate import dict_index
from utils.pickle_picky import load, save
from utils.randomness import set_global_random_seed
from mymodel.sccl import SCCL_BERT
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


# class Test_dataset(Dataset):
#     def __init__(self, dataset):
#         self.data = dataset

#     def __getitem__(self, index):
#         return {
#             'text': self.data['text'][index],
#             'label': self.data['label'][index]
#         }

#     def __len__(self):
#         return len(self.data['text'])


# class NLNL_Net(nn.Module):
#     def __init__(self,sccl_bert: SCCL_BERT):
#         super(NLNL_Net, self).__init__()
#         self.sccl_bert = sccl_bert

#     def forward(self, text_arr):
#         embd0 = self.sccl_bert.get_embeddings_PURE(text_arr)
#         out = self.sccl_bert.out(embd0)
#         return out



def NLNL_main(args):
    print(args)
    set_global_random_seed(args.seed)
    if args.train_path.find("wiki")!=-1:
        mode = "wiki"
    else:
        mode = "tac"
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

    # test_data = load(args.dev_path)
    # if mode=="tac":
    #     pos_index = [i for i,p_label in enumerate(test_data['label']) if p_label!=41]
    #     test_data = dict_index(test_data,pos_index)
    # if mode=="wiki":
    #     label2id = load("/home/tywang/myURE/URE/WIKI/typed/label2id.pkl")
    #     test_data['label'] = [label2id[item] for item in test_data['rel']]
    num_classes = args.n_rel
    args.N_train = N_train = len(train_data['text'])
    train_data['index'] = [i for i in range(args.N_train)]
    train_dataset = Train_dataset(train_data)
    # test_dataset = Test_dataset(test_data)
    train_loader = util_data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    # test_loader = util_data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
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
    # net:NLNL_Net = NLNL_Net(sccl_model).to(device)
    optimizer = torch.optim.AdamW([
        {'params': sccl_model.sentbert.parameters()},
        {'params': sccl_model.out.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)
    # optimizer = AdamW(model.parameters(), lr=2e-6, correct_bias=False) #  一般设定 4e-7 (succeeded experiment)
    ##

    # 产生weight
    weight = torch.FloatTensor(num_classes).zero_() + 1.
    for i in range(num_classes):
        weight[i] = (torch.from_numpy(np.array(train_data['p_label']).astype(int)) == i).sum()  # 首先计算各个类别分别有多少数据
    weight = 1 / (weight / weight.max())  # 然后再归一化???  没太懂为什么这里这么做



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

    ##
    # NLNL parameters
    train_preds = torch.zeros(N_train, num_classes) - 1.
    num_hist = 10
    train_preds_hist = torch.zeros(N_train, num_hist, num_classes)   # [45000,10,10]
    pl_ratio = 0.
    nl_ratio = 1.-pl_ratio  # ????
    train_losses = torch.zeros(N_train) - 1.  # 每个数据的loss
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
        for i, data in enumerate(train_loader):
            
            text, labels, index = data['text'],data['p_label'],data['index']
            labels_neg = (labels.unsqueeze(-1).repeat(1, args.ln_neg)
                      + torch.LongTensor(len(labels), args.ln_neg).random_(1, num_classes)) % num_classes
            assert labels_neg.max() <= num_classes-1
            assert labels_neg.min() >= 0
            assert (labels_neg != labels.unsqueeze(-1).repeat(1, args.ln_neg)
                    ).sum() == len(labels)*args.ln_neg  # 保证得到的都是和原来label不同的数据
            labels = labels.to(device)
            labels_neg = labels_neg.to(device)
            logits = sccl_model.out(sccl_model.get_embeddings_PURE(text))

            s_neg = torch.log(torch.clamp(1.-F.softmax(logits, -1), min=1e-5, max=1.))  # log(1-pk)
            s_neg *= weight[labels].unsqueeze(-1).expand(s_neg.size()).to(device)
            _, pred = torch.max(logits.data, -1)  # 预测值
            acc = float((pred == labels.data).sum())   # batch的正确个数
            train_acc += acc
            accs.append(acc/len(index))

            train_loss += logits.size(0)*criterion(logits, labels).data
            train_loss_neg += logits.size(0) * criterion_nll(s_neg, labels_neg[:, 0]).data

            train_losses[index] = criterion_nr(logits, labels).cpu().data  # 记录这次每个数据的 CEloss
            


            # if epoch >= args.switch_epoch1:  # ? 什么操作, 无论如何label都是-100
            #     if epoch>=args.switch_epoch2:
            #         labels[ train_preds_hist.mean(1)[index, labels] < args.cut ] = -100
            #         labels_neg = labels_neg*0 - 100  # 是不用NL的意思吗?
            #     else:
            #         if epoch == args.switch_epoch1 and i == 0:
            #             print('Switch to SelNL')
            #         labels_neg[train_preds_hist.mean(
            #             1)[index, labels] < 1/float(num_classes)] = -100
            #         labels = labels*0 - 100
            # else:
            #     labels = labels*0 - 100  # -100loss就会是0
            labels = labels*0 - 100  # In the program, we do not use the process of SelNL and SelPL, cause they will make lower accuracy in "clean data"
            
            loss_neg = criterion_nll(s_neg.repeat(args.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
            loss_pl = criterion(logits, labels)* float((labels >= 0).sum())
            
            loss = (loss_pl+loss_neg) / (float((labels >= 0).sum()) +float((labels_neg[:, 0] >= 0).sum()))
            loss.backward()
            optimizer.step()
            l = logits.size(0)*loss.detach().cpu().data
            train_loss+=l
            
            losses.append(l/logits.size(0))
            train_preds[index.cpu()] = F.softmax(logits, -1).cpu().data

            pl += float((labels >= 0).sum())
            print('\r', " step {}/{} ,  loss_{:.4f} acc_{:.4f}  ".format(i+1,len(train_loader),np.mean(losses),np.mean(accs)), end='', flush=True)
            nl += float((labels_neg[:, 0] >= 0).sum())
            # if i==10:break

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
        """先注释掉下面所有的代码, 只观察train模式"""
        # sccl_model.eval()
        # test_loss = test_acc = 0.0
        # with torch.no_grad():
        #     for i, data in enumerate(test_loader):
        #         text, labels = data['text'],data['label']
        #         logits =  sccl_model.out(sccl_model.get_embeddings_PURE(text))
        #         labels = Variable(labels.to(device))

        #         loss = criterion(logits, labels)
        #         test_loss += logits.size(0)*loss.data

        #         _, pred = torch.max(logits.data, -1)
        #         acc = float((pred == labels.data).sum())
        #         test_acc += acc
        # test_loss /= len(test_data['text'])
        # test_acc /= len(test_data['text'])



        inds = np.argsort(np.array(train_losses))[::-1] # 按照train_loss排序, index从大到小
        clean_index = np.argsort(np.array(train_losses))  # 按照loss从小到大的index

        ##
        # 画clean和noise数据的分布的图
        plt.hist(train_losses[inds_clean].numpy(), bins=33, edgecolor='black', alpha=0.5, label='clean', histtype='bar')
        plt.hist(train_losses[inds_noisy].numpy(), bins=33, edgecolor='black', alpha=0.5, label='noisy', histtype='bar')
        plt.xlabel('loss')
        plt.ylabel('number of data')
        plt.grid()
        plt.legend()
        plt.savefig(args.save_dir+'/{}_hist_LOSS_epoch{}_T{}.jpg'.format(mode,epoch,TIME))
        plt.clf()
    
        ##

        # indexes = []
        # select_ratio = [0.01,0.05,0.1,0.2,0.5,1.0]
        # # select_ratio = [0.07013,0.3507,0.701,1]
        # for select_rt in select_ratio:
        #     selected403  = clean_index[:int(N_train*select_rt)]
        #     indexes.append(selected403)
        #     Slabel = np.array([train_data['label'][index] for index in selected403])
        #     Sp_label = np.array([train_data['p_label'][index] for index in selected403])
        #     n_cate = len(set(Sp_label))
        #     acc = sum(Slabel==Sp_label)/len(Sp_label)
        #     # print("确定的acc:{:.4f}".format(acc))
        #     print("前 {} loss 小的数据(num:{})的acc= {}, 类别数:{}".format(select_rt,int(N_train*select_rt),acc,n_cate))
        # if epoch-1==args.epoch:
        #     save(indexes,os.path.join(CURR_DIR,"NLNL_index_out/{}_index_epo{}_acc{:.4f}_T{}.pkl".format(mode,epoch,acc,TIME)))
        rnge = int(N_train*noise_ratio)
        inds_filt = inds[:rnge]  # loss前N大的index
        recall = float(len(np.intersect1d(inds_filt, inds_noisy))) / float(len(inds_filt)) # 击中noisy的比例
        # precision = float(len(np.intersect1d(inds_filt, inds_noisy))) / float(rnge)
        # print('\tTESTING...loss: %5f, acc: %5f, best_acc: %5f, noisy_acc: %5f'
        #         % (test_loss, test_acc, best_test_acc, recall))
        # print(noise_ratio)
        # print(rnge)

        ###############################################################################################
        assert train_preds[train_preds < 0].nelement() == 0
        train_preds_hist[:, epoch % num_hist] = train_preds
        train_preds = train_preds*0 - 1.
        assert train_losses[train_losses < 0].nelement() == 0
        train_losses = train_losses*0 - 1.
        ###############################################################################################
        # is_best = test_acc > best_test_acc
        # best_test_acc = max(test_acc, best_test_acc)

        
        ##  
        # 计算仅仅由 本身confidence 排序得到的数据的acc
        p_label_confidence = train_preds_hist.mean(1)[torch.arange(N_train), np.array(train_data['p_label']).astype(int)] # shape = N_train
        confidence_index = np.argsort(np.array(p_label_confidence))[::-1]  # confidence从大到小排序
        indexes = []
        if mode=="wiki":
            select_num = [403,2016,4032,1e20] # number corresponding to(0.01, 0.05, 0.1, 1.0)*n_train (n_train=40320)
        else:
            select_num = [681,3406,6812,1e20] # number corresponding to(0.01, 0.05, 0.1, 1.0)*n_train (n_train=68124)
        for select_n in select_num:
            selected403  = confidence_index[:int(select_n)]
            indexes.append(selected403)
            Slabel = np.array([train_data['label'][index] for index in selected403])
            Sp_label = np.array([train_data['p_label'][index] for index in selected403])
            n_cate = len(set(Sp_label))
            acc = sum(Slabel==Sp_label)/len(Sp_label)
            # print("确定的acc:{:.4f}".format(acc))
            print("前 {} confidence 大的数据 acc= {}, 类别数:{}".format(select_n,acc,n_cate))
        if epoch+1==args.epoch:
            # select 数据
            ratio = [0.01,0.05,0.1]
            for index,rt in zip(indexes[:3],ratio):
                selected_data = dict_index(train_data,index)
                acc = sum(np.array(selected_data['label'])==np.array(selected_data['top1']))/len(selected_data['label'])
                print("selected data acc:{}".format(acc))
                save(selected_data,os.path.join(PROJECT_PATH,"finetune/selected_data/{}/selected_n{}train.pkl".format(mode,rt)))
            #save(indexes,os.path.join(CURR_DIR,"NLNL_index_out/{}_index_epo{}_acc{:.4f}_T{}_CONFIDENCE.pkl".format(mode,epoch,acc,TIME)))

        ##

        if epoch %1 == 0 :
            print('saving histogram...')
            g_data1 = p_label_confidence
            g_data1 = g_data1.numpy()
            plt.hist(g_data1, bins=33, range=(0, 1),
                    edgecolor='black', histtype='bar')
            plt.xlabel('probability')
            plt.ylabel('number of data')
            plt.grid()
            plt.savefig(args.save_dir+'/{}_histogram_epoch{}_T{}.jpg'.format(mode,epoch,TIME))
            plt.clf()
            print('saving separated histogram...')
            g_data2 = train_preds_hist.mean(1)[torch.arange(N_train)[
                inds_clean], np.array(train_data['p_label']).astype(int)[inds_clean]]
            g_data3 = train_preds_hist.mean(1)[torch.arange(N_train)[
                inds_noisy], np.array(train_data['p_label']).astype(int)[inds_noisy]]
            g_data2 = g_data2.numpy()
            g_data3 = g_data3.numpy()
            # train_preds_hist.mean(1)[torch.arange(N_train)[inds_clean], np.array(train_data['p_label'])[:,1].astype(int)[inds_clean]] 每个数据都有的一个confidence值
            plt.hist(g_data2, bins=33, edgecolor='black', alpha=0.5,
                    range=(0, 1), label='clean', histtype='bar')
            plt.hist(g_data3, bins=33, edgecolor='black', alpha=0.5,
                    range=(0, 1), label='noisy', histtype='bar')
            plt.xlabel('probability')
            plt.ylabel('number of data')
            plt.grid()
            plt.legend()
            plt.savefig(args.save_dir+'/{}_histogram_sep_epoch{}_T{}.jpg'.format(mode,epoch,TIME))
            plt.clf()
            # np.save("/home/tywang/myURE/URE/NLNL_bert/NLNL_out/NLNL_plot_data" +
            #         '/plot_data%03d.npy' % (epoch), [g_data1, g_data2, g_data3], allow_pickle=True)





if __name__ == "__main__":
    """
cd /home/tywang/myURE/URE/NLNL_bert
nohup python -u NL_v2.py >/home/tywang/myURE/URE/NLNL_bert/logs/xxx.log 2>&1 &
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=16, help="as named")
    parser.add_argument("--cuda_index", type=int, default=0, help="as named")
    parser.add_argument('--lr', type=float, default=5e-7, help='learning rate')
    parser.add_argument('--lr_scale', type=int, default=100, help='as named')
    parser.add_argument('--epoch', type=int, default=1, help='as named')
    # parser.add_argument('--cut', type=float, default=0.5, help='as named')
    # parser.add_argument('--switch_epoch1', type=int, default=20, help='as named')
    # parser.add_argument('--switch_epoch2', type=int, default=25, help='as named')

    """wiki"""
    # parser.add_argument('--n_rel', type=int, default=80, help='as named')
    # parser.add_argument("--train_path", type=str,
    #                     default="/home/tywang/myURE/URE_mnli/temp_files/analysis_0.01510/wiki_train_num40320_top1_0.4112_Xlarge.pkl", help="as named")
    # # parser.add_argument("--dev_path", type=str,
    # #                     default="/home/tywang/myURE/URE/WIKI/typed/wiki_devwithtype_premnil.pkl", help="as named")
    # parser.add_argument("--e_tags_path", type=str,
    #                     default="/home/tywang/myURE/URE/WIKI/typed/etags.pkl", help="as named")
    # parser.add_argument("--save_dir", type=str,
    #                     default="/home/tywang/URE-master/NLNL_bert/NLNL_out", help="it is used to save imgs")
    # parser.add_argument('--ln_neg', type=int, default=80,
    #                     help='number of negative labels on single image for training, equal to n_rel')

    """tac"""
    parser.add_argument('--n_rel', type=int, default=41, help='as named')
    parser.add_argument("--train_path", type=str,
                        default="/home/tywang/myURE/URE_mnli/temp_files/analysis_0.01510/tac_NLNL_num9710_acc0.5799_allpos.pkl", help="as named")
    # parser.add_argument("--dev_path", type=str,
    #                     default="/home/tywang/myURE/URE/O2U_bert/tac_data/whole/test_for_top12.pkl", help="as named")
    parser.add_argument("--e_tags_path", type=str,
                        default="/home/tywang/myURE/URE/O2U_bert/tac_data/train_tags.pkl", help="as named")
    parser.add_argument("--save_dir", type=str,
                        default="/home/tywang/URE-master/NLNL_bert/NLNL_out", help="as named")
    parser.add_argument('--ln_neg', type=int, default=41,
                        help='number of negative labels on single image for training (ex. 110 for cifar100)')


    """communal"""
    parser.add_argument('--model_dir', type=str,
                        default='/data/transformers/bert-base-uncased', help='as named')
    parser.add_argument('--max_len', type=int, default=64,
                        help='length of input sentence')
    args = parser.parse_args()
    NLNL_main(args)

    