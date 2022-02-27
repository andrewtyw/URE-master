import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np
import random
eps = 1e-8
device = torch.device('cuda:0')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# contrast_loss
# 仅仅根据两个B的embedding就可以计算出contrast loss

# def get_pair_keyword(batch_prob, feat, k):
#     """
#
#     :param batch_prob: 每个batch pro 的分布
#     :param feat: [bs,sen_dim] 的张量
#     :param k:
#     :param device:
#     :return:
#     """
#     couple = []
#     L = len(batch_prob)
#     similarity = torch.zeros([L, L])
#     for i in range(L):
#         similarity[i, :] = torch.cosine_similarity(batch_prob[i].unsqueeze(0), batch_prob)
#         similarity[i][i] = 0
#         if similarity[i, :].max().detach() == 1:
#             index = torch.where(similarity[i, :] == 1)[0].detach().cpu().numpy()
#             # return index
#             # print(np.array(index))
#             if len(index) < k - 1:
#                 couple.append(feat[torch.topk(similarity[i], k - 1)[1]])
#             else:
#                 choices = torch.from_numpy(np.random.choice(np.array(index), k - 1))
#
#                 couple.append(feat[choices])
#         else:
#             couple.append(feat[torch.topk(similarity[i], k - 1)[1]])
#     #         print(feat[torch.topk(similarity[i],k-1)[1]].shape)
#     res = torch.cat((feat.unsqueeze(1), torch.stack(couple)), dim=1)
#     return res, similarity


def get_pair(batch_prob, feat, k):
    """

    :param batch_prob: 每个batch pro 的分布
    :param feat: [bs,sen_dim] 的张量
    :param k:
    :param device:
    :return:
    """
    couple = []
    L = len(batch_prob)
    similarity = torch.zeros([L, L])
    for i in range(L):
        similarity[i, :] = torch.cosine_similarity(batch_prob[i].unsqueeze(0), batch_prob)
        similarity[i][i] = 0
        couple.append(feat[torch.topk(similarity[i], k - 1)[1]])
    #         print(feat[torch.topk(similarity[i],k-1)[1]].shape)
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
    #         print(feat[torch.topk(similarity[i],k-1)[1]].shape)
    # res = torch.cat((feat.unsqueeze(1),torch.stack(couple)),dim = 1)
    # res = torch.cat((feat.unsqueeze(1), torch.stack(couple)), dim=1)
    return couple, similarity

def get_pair_max(batch_prob, feat, k, device):
    couple = []
    L = len(batch_prob)
    similarity = torch.zeros([L, L])
    for i in range(L):
        similarity[i, :] =  torch.cosine_similarity(batch_prob[i].unsqueeze(0), batch_prob)
        similarity[i][i] = 0
        couple.append(feat[torch.topk(similarity[i], k - 1)[1]])
    #         print(feat[torch.topk(similarity[i],k-1)[1]].shape)
    # res = torch.cat((feat.unsqueeze(1),torch.stack(couple)),dim = 1)
    # res = torch.cat((feat.unsqueeze(1), torch.stack(couple)), dim=1)
    return couple, similarity


# def get_pair(batch_prob,feat,device):
#     couple = []
#     L = len(batch_prob)
#     similarity = torch.zeros([L,L])
#     for i in range(L):
#         similarity[i,:] = torch.cosine_similarity(batch_prob[i].unsqueeze(0),batch_prob)
#         similarity[i][i] = 0
#         couple.append(feat[torch.argmax(similarity[i])])
#     return torch.stack(couple).to(device),similarity

class MaskLoss(nn.Module):
    def __init__(self, topk=10):
        super(MaskLoss, self).__init__()
        self.topk = topk

    def forward(self, pre_label_masks, pre_label):
        """
        计算label之间的误差
        :param pre_label_masks: object或者subject换了之后的数据的预测
        :param pre_label: 真正数据的预测
        :return:
        """
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
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, device=torch.device('cuda:0')):
        """
        :param temperature:  t
        :param contrast_mode:
        :param base_temperature:
        """
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss
        :param features:  B^alpha 中的2个augment sentence拼接而成,[Batch_size,2*S_L,d_model] 三维
        :return:
        """
        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (Batch_size+2*S_L,)

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

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    用于计算pjk
    :param batch: qjk
    :return:
    """
    weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)  # 分子
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
