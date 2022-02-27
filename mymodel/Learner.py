import sys
import os
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from URE.mymodel.sccl import SCCL_BERT
from URE.utils.loss_utils import SupConLoss,KCL,target_distribution,DispersionLoss
# sys.path.append()
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件夹

class ClusterLearner(nn.Module):
    def __init__(self, sccl_pcnn:SCCL_BERT, optimizer, temperature, base_temperature,device):
        super(ClusterLearner, self).__init__()
        self.device = device
        self.SCCL_PCNN = sccl_pcnn
        self.optimizer = optimizer
        #MaskLoss
        self.Dloss = DispersionLoss()
        # self.contrast_loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        # self.prob_loss = SupConLoss(temperature=temperature, base_temperature=base_temperature,contrast_mode='one')
        # self.cluster_loss = nn.KLDivLoss(reduction='sum')  # KL散度损失 D(P||Q) = sum P(x)log(P(x)/Q(x))
        self.CEloss = nn.CrossEntropyLoss()
        self.kcl = KCL() 

    def forward(self, inputs,i_epoch=None,use_perturbation=False,use_dloss = False):

        # 找出有pseudo label的下标
        if isinstance(inputs[3],list):
            p_rel = torch.tensor(inputs[3]).long().to(self.device)
        else:
            p_rel = inputs[3].to(self.device)
        p_index = torch.where(p_rel>=0)[0] #有pesudo label的index
        p_rel = p_rel[p_index]


        ##
        #这里的input是个list，里面是文本
        ##

        embd0 = self.SCCL_PCNN.get_embeddings(inputs[0])
        ##
        # embd1 = self.SCCL_PCNN.get_embeddings(inputs[1])
        # embd2 = self.SCCL_PCNN.get_embeddings(inputs[2])


        # # Instance-CL loss
        # feat1 = F.normalize(self.SCCL_PCNN.head(embd1), dim=1)  # head embed_size->128 线性层
        # feat2 = F.normalize(self.SCCL_PCNN.head(embd2), dim=1)

        # features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)

        ##
        
        #print(features.shape)
        # contrastive_loss = self.contrast_loss(features)  # 得到一个三维向量，计算contrast_loss
        contrastive_loss = torch.tensor(0)


        p_loss = torch.tensor(0)
        D_loss = torch.tensor(0)

        #CE loss
        embed_pseudo = embd0[p_index]
        out = self.SCCL_PCNN.out(embed_pseudo)
        
        p_loss = self.CEloss(out,p_rel)
        if use_dloss:
            out_soft = F.softmax(out,dim=-1)
            D_loss = self.Dloss(out_soft)

        #clustring loss
        # embed0_f = F.normalize(self.SCCL_PCNN.f(embd0), dim=1)
        # output = self.SCCL_PCNN.get_cluster_prob(embed0_f)  # 得到qjk
        # target = target_distribution(output).detach()  # 得到pjk
        # cluster_loss = (self.cluster_loss((output + 1e-08).log(), target) / output.shape[0])*10#*yita

        cluster_loss = torch.tensor(0)
        

        #loss =  contrastive_loss+cluster_loss+p_loss
        loss =  p_loss+D_loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"Instance-CL_loss": contrastive_loss.detach(), "clustering_loss": D_loss.detach(),'loss3':p_loss}