import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flair.models import SequenceTagger
from flair.data import Sentence
import glob
import os
from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np
import string
import argparse
from tqdm import tqdm
import copy
from collections import Counter
import random
import math
from os import listdir
from os.path import isfile, join, exists
import json
import copy
from torch.utils.data import DataLoader, Dataset
import pickle

def save(obj,path_name):
    with open(path_name,'wb') as file:
        pickle.dump(obj,file)

def load(path_name: object) :
    with open(path_name,'rb') as file:
        return pickle.load(file)

tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

def findEntity(text:str):
    sentence = Sentence(text)
    tagger.predict(sentence)
    tmp_dict={}
    for entity in sentence.to_dict(tag_type='ner')['entities']:
        tmp_dict[entity['text'].lower()]=entity['labels'][0].to_dict()['value']
    return tmp_dict


def process_data(args):

    cnt_of_single=0
    cnt_of_double=0
    cnt_of_sample=0
    data=load(args.datapath)
    extended_data=[]
    for index,d in enumerate(data): #238212

        if(index%1000==0):
            print("{}:{}".format(index,len(data)))
            print('cnt_of_single:{}/cnt_of_double:{}'.format(cnt_of_single,cnt_of_double))
            print('cnt_of_smaple:{}'.format(cnt_of_sample))
            
        if str(d['text']).split('.').__len__()<3:
            cnt_of_single+=1 
            continue
        else:
            cnt_of_double+=1
            sents=d['text'].split('.')
            for sent in sents[0:1]:
                if sent!="":
                    entitys=findEntity(sent)
                    for i,j in entitys.items():
                        cnt_of_sample+=1
                        extended_data.append({'text':'. '.join([sents[0],sent]),'subj':d['subj']
                        ,'type':d['type'],
                        'entitys':i,
                        'entityType':j})
                        break

    save(extended_data,args.savepath)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, required=True,
                        help='dbpedia data to type')
    parser.add_argument('--savepath', type=str, required=True,
                        help='the path to save typed data')
    
    args = parser.parse_args()

    process_data(args)
    
