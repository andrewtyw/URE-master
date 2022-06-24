import copy
from os import error
import sys
import numpy as np
import torch


def dict2list(data:dict):
    keys = list(data.keys())
    assert_set = set()
    for k in keys:
        assert_set.add(len(data[k]))
    assert len(assert_set)==1
    items = []
    for i in range(len(data[k])):
        item = {}
        for k in keys:
            item[k] = data[k][i]
        items.append(item)
    return items

def dict_index(data:dict,index_arr):

    
    length = len(data[list(data.keys())[0]])

    data_copy = copy.deepcopy(data)
    for k,_ in data_copy.items():
        l = len(data_copy[k])
        if l!=length:
            print()
            print()
            print("***************************************")
            print("*       key:{} len is {}, skip!       *".format(k, l))
            print("***************************************")
            print()
            print()
            continue

        

        if isinstance(data_copy[k],np.ndarray):
            data_copy[k] = np.array([data_copy[k][idx] for idx in index_arr])
        elif isinstance(data_copy[k],torch.Tensor):
            data_copy[k] = data_copy[k][torch.tensor(index_arr).long()]
        else: 
            data_copy[k] = [data_copy[k][idx] for idx in index_arr]


        
    return data_copy


if __name__=="__main__":
    pass
