import torch
from URE.mymodel.sccl import SCCL_BERT
from URE.mymodel.Learner import ClusterLearner
from URE.mymodel.dataset import get_eval_loader

# from sklearn.metrics import accuracy_score,recall_score,precision_score
import sklearn.metrics as metrics
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score
from URE.mymodel.sccl import SCCL_BERT


# def evaluation_softmax(cluster_learner,dataset):
#     with torch.no_grad():
#         cluster_learner.SCCL_PCNN.eval()
#         Out = cluster_learner.SCCL_PCNN.out
#         embed = cluster_learner.SCCL_PCNN.get_embeddings(dataset['text'])
#         res = Out(embed)
#         # print(res.shape)
        
#         y_pre = torch.max(res,dim=-1)[1].detach().cpu().numpy().tolist()
#         y_true = dataset['label']
#         f1 = cal_f1(y_true,y_pre)[0]
#         return f1
    
def evaluation_softmax_loader(cluster_learner:ClusterLearner,eval_loader,n_rel):
    with torch.no_grad():
        cluster_learner.SCCL_PCNN.eval()
        Out = cluster_learner.SCCL_PCNN.out
        Y_pre = []
        Y_true = []
        Res = [] # 也就是一个 n x n_rel 的矩阵
        for batch in eval_loader:
            embed = cluster_learner.SCCL_PCNN.get_embeddings(batch['text'])

            res = Out(embed)
            #所有的结果经过softmax
            res = F.softmax(res,dim=-1)
            
            Res.append(res)
            # print(res.shape)
            y_pre = torch.max(res,dim=-1)[1].detach().cpu().numpy().tolist()
            Y_pre+=y_pre
            Y_true+=batch['label'].numpy().tolist()
        # f1,dicts,tru, y_pre_trans = cal_f1(Y_pre,Y_true,n_rel)

        # !!
        f1 = f1_score(Y_true, Y_pre, average='macro')
        # y_pre_trans = Y_pre
        dicts = None
        

        Res = torch.cat(Res).detach().cpu()
        # print(Res.shape)

    return f1,Res,Y_true,Y_pre,dicts

def evaluation_softmax_loader_uda(cluster_learner,eval_loader,n_rel,return_pre=False,return_confident = False):
    with torch.no_grad():
        cluster_learner.sccl_bert.eval()
        Out = cluster_learner.sccl_bert.out
        Y_pre = []
        Y_true = []
        Res = [] # 也就是一个 n x n_rel 的矩阵
        confidence = []
        for batch in tqdm(eval_loader):
            embed = cluster_learner.sccl_bert.get_embeddings_PURE(batch['text'])

            res = Out(embed)
            #所有的结果经过softmax
            res = F.softmax(res,dim=-1)
            
            Res.append(res)
            # print(res.shape)
            y_pre = torch.max(res,dim=-1)[1].detach().cpu().numpy().tolist()
            confident = torch.max(res,dim=-1)[0].detach().cpu().numpy().tolist()
            confidence+=confident
            Y_pre+=y_pre
            Y_true+=batch['label'].numpy().tolist()
        # f1 = cal_f1(Y_true,Y_pre,n_rel)[0]
        f1 = f1_score(Y_true, Y_pre, average='macro')
        accuracy = metrics.accuracy_score(Y_true, Y_pre)
        recall = metrics.recall_score(Y_true, Y_pre, average="macro")
        precision = metrics.precision_score(Y_true, Y_pre, average="macro",zero_division=0)
        metric = [accuracy,recall,precision,f1]
        Res = torch.cat(Res).detach().cpu()
        # print(Res.shape)
    if return_pre:
        return Y_pre,metric
    if return_confident:
        return Y_true,Y_pre,confidence,metric
    return f1,Res,Y_true,metric

def predict(cluster_learner:ClusterLearner,eval_dict:dict,n_rel,return_tensor=True):
    """
    给一个dict,它要包含text和label这两个key,然后生成一个loader(为了减少eval的时候显存的占用)
    """
    eval_loader = get_eval_loader(eval_dict)
    # f1,Res,Y_true,dicts 
    f1,Res,Y_true,Y_pre,dicts= evaluation_softmax_loader(cluster_learner,eval_loader,n_rel)

    # print(Y_true)
    # print(Y_pre)
    accuracy = metrics.accuracy_score(Y_true, Y_pre)
    recall = metrics.recall_score(Y_true, Y_pre, average="macro")
    precision = metrics.precision_score(Y_true, Y_pre, average="macro",zero_division=0)
    # F1 = metrics.f1_score(Y_true, Y_pre, average="macro") 
    metric = [accuracy,recall,precision,f1]
    if return_tensor:
        return f1,Res,Y_true,metric
    else:
        Res = Res.numpy()
        return f1,Res,Y_true,metric
        


