from torch.utils.data import Dataset, dataset
import torch.utils.data as util_data
import os
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件夹

class datasets_train(Dataset):
    def __init__(self, train_dataset) -> object:
        self.text = train_dataset['text']
        self.text1 = train_dataset['text1']
        self.text2 = train_dataset['text2']
        self.train_y = torch.tensor(train_dataset['label']).long()
        self.p_train_y = torch.tensor(train_dataset['p_label']).long()
        # self.prob = train_dataset['prob'] # 老师推荐的监督

    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        return {
            'text': self.text[idx],
            'text1': self.text1[idx],
            'text2': self.text2[idx],
            'label': self.train_y[idx],
            'p_label': self.p_train_y[idx]}

class datasets_dev_test(Dataset):
    def __init__(self, train_dataset) -> object:
        self.text = train_dataset['text']
        if isinstance(train_dataset['label'],torch.Tensor):
            self.train_y = train_dataset['label'].clone().detach()
        else:  
            self.train_y = torch.tensor(train_dataset['label']).long()
        # self.prob = train_dataset['prob'] # 老师推荐的监督

    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        return {
            'text': self.text[idx],
            'label': self.train_y[idx]}

def get_train_loader(train_data:dict,batch_size = 384):
    dataset = datasets_train(train_data)
    train_loader = util_data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader

def get_eval_loader(eval_data:dict,batch_size = 384):
    """
    得到验证的数据集的loader, 不然太多数据一次过eval的话太耗显存
    """
    dataset = datasets_dev_test(eval_data)
    eval_loader = util_data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return eval_loader