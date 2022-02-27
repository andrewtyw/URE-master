import random
import pickle
import argparse
import torch
import copy
import numpy as np
random.seed(16) #随机数粽子

def dict_index(data:dict,index_arr):
    """
    遍历dict中的元素, 取出在index_arr中的下标
    """
    data_copy = copy.deepcopy(data)
    for k,_ in data_copy.items():
        if isinstance(data_copy[k],np.ndarray):
            data_copy[k] = np.array([data_copy[k][idx] for idx in index_arr])
        elif isinstance(data_copy[k],torch.Tensor):
            data_copy[k] = data_copy[k][torch.tensor(index_arr).long()]
        else: 
            data_copy[k] = [data_copy[k][idx] for idx in index_arr]
    return data_copy


def save(obj,path_name):
    with open(path_name,'wb') as file:
        pickle.dump(obj,file)

def load(path_name: object) -> object:
    with open(path_name,'rb') as file:
        return pickle.load(file)

def get_train_and_dev(args):
    train = load(args.train_path)
    L = len(train['text'])
    index = [i for i in range(L)]
    random.shuffle(index)
    num_dev = int(L*0.2)
    dev_index = index[:num_dev]
    train_index = index[num_dev:]
    Dev = dict_index(train,dev_index)
    Train = dict_index(train,train_index)
    return Dev,Train




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default="/root/tywang/code_for_infer_T5/data/wiki80/train.pkl", help='as named')
    args = parser.parse_args()
    train,dev = get_train_and_dev(args)  # 可以保存它了

