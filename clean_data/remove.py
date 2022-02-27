import os
from pickle import decode_long
import sys
import copy
import re
import random
os.chdir(sys.path[0])
sys.path.append('/root/tywang/URE')
from utils.pickle_picky import *




def remove_etype(datas:dict,text_k='text',tags = load("/root/tywang/URE/data/tacred/tacred_e_tag/tacred_e_tag.pkl")):
    tags = [re.findall(r':(.*?)>',tag)[0] for tag in tags]
    data = copy.deepcopy(datas)
    texts = []
    for text in datas[text_k]:
        for tag in tags:
            text = text.replace(tag,"")
        texts.append(text)
    data[text_k] = texts
    return data

def remove_front_latter(datas:dict,text_k='text'):
    data = copy.deepcopy(datas)
    texts = []
    for text in datas[text_k]:
        text = re.findall(r'<.*>',text)[0]
        texts.append(text)
    data[text_k] = texts
    return data

if __name__=="__main__":
    # tags = load("/root/tywang/URE/data/tacred/tacred_e_tag/tacred_e_tag.pkl")
    # tags = [re.findall(r':(.*?)>',tag)[0] for tag in tags]
    # print(tags)
    datas = load("/root/tywang/URE/data/tacred/train.pkl")
    print(datas.keys())
    texts = remove_etype(datas)
    for t1,t2 in zip(datas['text'],texts['text']):
        if random.random()<0.01:
            print("===========")
            print(t1)
            print(t2)


