
import sys
import os
from pathlib import Path

from sklearn import metrics

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
import random
import re
from collections import Counter

def merge(args):
    print(args)
    inject_data = load(os.path.join(CURR_DIR,"annotated_extra_data/{}_inject.pkl".format(args.dataset)))
    target = load(args.clean_data_path)
    try:
        del target['score']
    except:
        pass
    assert inject_data.keys()==target.keys()
    print("len target=",len(target['text']))
    print("len inject=",len(inject_data['text']))

    for k in target.keys():
        target[k].extend(inject_data[k])
        print(k," ",len(target[k]))
    save(target,args.save_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="", help="as named")
    parser.add_argument("--clean_data_path", type=str,default="", help="as named")
    parser.add_argument("--save_path", type=str,default="", help="as named")
    args = parser.parse_args()

    merge(args)
