import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np
import random
eps = 1e-8
import re
import sys
device = torch.device('cuda:0')


def get_pair(batch_prob, feat, k):

    couple = []
    L = len(batch_prob)
    similarity = torch.zeros([L, L])
    for i in range(L):
        similarity[i, :] = torch.cosine_similarity(batch_prob[i].unsqueeze(0), batch_prob)
        similarity[i][i] = 0
        couple.append(feat[torch.topk(similarity[i], k - 1)[1]])
    
    res = torch.cat((feat.unsqueeze(1), torch.stack(couple)), dim=1)
    return res, similarity


def get_pair_min(batch_prob, feat, k, device):
    couple = []
    L = len(batch_prob)
    similarity = torch.zeros([L, L])
    for i in range(L):
        similarity[i, :] = 1 / torch.cosine_similarity(batch_prob[i].unsqueeze(0), batch_prob)
        similarity[i][i] = 1
        couple.append(feat[torch.topk(similarity[i], k - 1)[1]])
    
    
    
    return couple, similarity

def get_pair_max(batch_prob, feat, k, device):
    couple = []
    L = len(batch_prob)
    similarity = torch.zeros([L, L])
    for i in range(L):
        similarity[i, :] =  torch.cosine_similarity(batch_prob[i].unsqueeze(0), batch_prob)
        similarity[i][i] = 0
        couple.append(feat[torch.topk(similarity[i], k - 1)[1]])
    
    
    
    return couple, similarity












class MaskLoss(nn.Module):
    def __init__(self, topk=10):
        super(MaskLoss, self).__init__()
        self.topk = topk

    def forward(self, pre_label_masks, pre_label):

        pre = pre_label.repeat(self.topk, 1).T.ravel()
        return torch.count_nonzero(pre_label_masks - pre) / pre_label.shape[0]

class DispersionLoss(nn.Module):
    def __init__(self):
        super(DispersionLoss,self).__init__()
    def forward(self,out):
        avg = out.mean(0)
        loss_d = (avg * torch.log(avg + 1e-5)).sum()
        return loss_d

class SupConLoss(nn.Module):
    def __init__(self,args, temperature=0.07, contrast_mode='one', base_temperature=0.07):
        """
        :param temperature:  t
        :param contrast_mode:
        :param base_temperature:
        """
        super(SupConLoss, self).__init__()
        self.device = torch.device("cuda:{}".format(args.cuda_index))
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss
        :param features:  B^alpha 
        :return:
        """
        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            print(mask)
            mask = mask.float().to(self.device)

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        
        mask = mask.repeat(anchor_count, contrast_count)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        
        exp_logits = torch.exp(logits) * logits_mask  
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        nonreduce_loss = loss.clone().detach().cpu()
        
        loss = loss.view(anchor_count, batch_size).mean()

        return loss,nonreduce_loss


def target_distribution(batch: torch.Tensor) -> torch.Tensor:

    weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)  
    return (weight.t() / torch.sum(weight, 1)).t()


class KLDiv(nn.Module):
    def forward(self, predict, target):
        assert predict.ndimension() == 2, 'Input dimension must be 2'
        target = target.detach()
        p1 = predict + eps
        t1 = target + eps
        logI = p1.log()
        logT = t1.log()
        TlogTdI = target * (logT - logI)
        kld = TlogTdI.sum(1)
        return kld


class KCL(nn.Module):
    def __init__(self):
        super(KCL, self).__init__()
        self.kld = KLDiv()

    def forward(self, prob1, prob2):
        kld = self.kld(prob1, prob2)
        return kld.mean()


class Augmentation:
    """
        This class is functioned as generating test augmentation by replacing the subject or 
        object to other entity with the same type. 
        For example,
    """
    def __init__(self,data):
        self.Dict = self.__get_subj_obj_dict(data)

    def get_augs(self,texts,k):
        """
        Args:
            texts: List[str], i.e. [x1,x2,x3..] where xi is a sentence
            k: int, the number of augmentation need to be generate for each sentence
        Return:
            List[List[str]], i.e.
                [[x1', x2', x3' ...],
                 [x1'',x2'',x3''...],
                 [...]]
                 where x1' and x1'' are diffenent augmentation of x1.
        """
        all_new_texts = [list() for _k in range(k)]
        for text in texts:
            new_texts = self.get_aug(text,k)
            for i in range(k):
                all_new_texts[i].append(new_texts[i])
        return all_new_texts
    def __get_subj_obj_dict(self,data):
        Dict = {'subject':dict(), 'object':dict()}
        all_subj_type = list(set(data['subj_type']))
        all_obj_type = list(set(data['obj_type']))
        self.all_subj_type = all_subj_type
        self.all_obj_type = all_obj_type
        SS = "<S:("+"|".join(all_subj_type)+")>(.*)</S:.*>"
        SO = "<O:("+"|".join(all_obj_type)+")>(.*)</O:.*>"
        CPS = re.compile(SS)
        CPO = re.compile(SO)
        self.CPS = CPS
        self.CPO = CPO
        for subj,subj_type, obj,obj_type in zip(
            data['subj'], data['subj_type'],data['obj'], data['obj_type'] 
        ):
            if subj_type not in Dict['subject']:
                Dict['subject'][subj_type] = [subj]
            else:
                Dict['subject'][subj_type].append(subj)
            
            if obj_type not in Dict['object']:
                Dict['object'][obj_type] = [obj]
            else:
                Dict['object'][obj_type].append(obj)
        return Dict
    def get_aug(self,text,k = 1):

        try:
            subj_search_res = self.CPS.search(text)
            obj_search_res = self.CPO.search(text)
            try:
                subj_start, subj_end = subj_search_res.span(2) 
                parse_subjtype, subj = subj_search_res.groups()
                assert parse_subjtype.strip() in self.all_subj_type
            except:
               subj_start, subj_end =  (None,None)
               parse_subjtype, subj = (None, None)
            try:
                obj_start, obj_end = obj_search_res.span(2)
                parse_objtype, obj = obj_search_res.groups()
                assert parse_objtype.strip() in self.all_obj_type
            except:
                obj_start, obj_end = (None, None)
                parse_objtype, obj =  (None, None)
            
            assert subj_start is not None or obj_start is not None
            
        except:
            print(text)
            sys.exit()

        if subj_start is None or obj_start is None:
            if subj_start is None:
                try:
                    random_entity = random.choices(self.Dict['object'][parse_objtype],k=k)
                except:
                    print(text)
                    sys.exit()
                start,end = obj_start, obj_end
            else:
                try:
                    random_entity = random.choices(self.Dict['subject'][parse_subjtype],k=k)
                except:
                    print(text)
                    sys.exit()
                start,end = subj_start, subj_end
        else:
            rn = random.random()
            if rn>0.5:
                
                try:
                    random_entity = random.choices(self.Dict['subject'][parse_subjtype],k=k)
                except:
                    print(text)
                    sys.exit()
                start,end = subj_start, subj_end
            else:
                
                try:
                    random_entity = random.choices(self.Dict['object'][parse_objtype],k=k)
                except:
                    print(text)
                    sys.exit()
                start,end = obj_start, obj_end
        texts = []
        for entity in random_entity:
            aug_text = text[:start]+" "+entity+" "+text[end:]
            texts.append(aug_text)
        return texts
