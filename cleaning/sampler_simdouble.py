
import argparse
from cProfile import label
import sys
import os
from pathlib import Path
import time
TIME=time.strftime("%m%d%H%M%S", time.localtime())

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

from utils.pickle_picky import load, save
import numpy as np
from collections import Counter
from utils.dict_relate import dict_index


def sampler(args):

    ratios = [0.01,0.21,0.28,0.35,0.25,0.105,0.075,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.30]

    if args.dataset=='tac':
        n_rel=42
        n_train=68124
        train_data=load('/data/tywang/URE_share/data/tac/tac_pseudo_pos.pkl')
        # train_data=load('/data/tywang/URE_share/data/tac/train_annotated.pkl')
        train_data['p_label'] = train_data['top1']

        label2id=load('/home/jwwang/URE_share/data/tac/label2id.pkl')
        id2label={}
        for key,value in label2id.items():
            id2label[value]=key

        confidence_path='/home/jwwang/URE_share/outcome/NL/clean_data/tac_seed{}_pconfidence.pkl'.format(args.seed)
        p_label_confidence=load(confidence_path)
        confidence_index = np.argsort(np.array(p_label_confidence))[::-1]
    elif args.dataset=="wikifact":
        label2id=load('/home/jwwang/URE_share/data/tac/label2id.pkl')
        id2label={}
        for key,value in label2id.items():
            id2label[value]=key
        whole_data=load('/home/jwwang/URE_share/outcome/NL/wikifact4tac/wikifact4tac_whole.pkl')
        train_data={
            'text':[],
            'label':[],
            'p_label':[],
            'top1':[],
            'subj':[],
            'obj': [],
            'template': []
        }
        keyMap={
            'text':'text',
            'label': 'target',
            'p_label':'p_rel',
            'top1':'top1',
            'subj': 'subj',
            'obj': 'obj',
            'template': 'template'
        }
        p_label_confidence=[]
        for item in whole_data:
            item['p_rel']=label2id[item['p_rel']]
            item['target']=item['p_rel']
            for key in train_data.keys():
                train_data[key].append(item[keyMap[key]])
            p_label_confidence.append(item['confidence'])
        confidence_index = np.argsort(np.array(p_label_confidence))[::-1]
    elif args.dataset=="wiki":
        n_rel=80
        n_train = 40320
        train_data=load('/home/jwwang/URE_share/data/wiki/train_annotated.pkl')
        train_data['p_label'] = train_data['top1']

        label2id=load('/home/jwwang/URE_share/data/wiki/label2id.pkl')
        id2label={}
        for key,value in label2id.items():
            id2label[value]=key

        confidence_path='/home/jwwang/URE_share/outcome/NL/clean_data/wiki/wiki_seed{}_pconfidence.pkl'.format(args.seed)
        p_label_confidence=load(confidence_path)
        confidence_index = np.argsort(np.array(p_label_confidence))[::-1]


    threshold_confidence=p_label_confidence[confidence_index[int(0.01*n_train)]]
    print('=> 0.01part of train is [{}]'.format(threshold_confidence))


    indexes = {}
    select_num = [int(rt*n_train) for rt in ratios]
    accs = []
    for select_n,rt in zip(select_num,ratios):
        selected403  = confidence_index[:int(select_n)]
        indexes[rt]=selected403

        # 计算acc
        text= np.array([train_data['text'][index] for index in selected403])
        Slabel = np.array([train_data['label'][index] for index in selected403]) # ground-truth
        Sp_label = np.array([train_data['p_label'][index] for index in selected403]) # pseudo label
        counter=Counter()
        for item in Sp_label:
            counter[item]+=1
        counter=sorted(counter.items(),key=lambda x:x[1],reverse=True)
        print('*'*50)
        for key,val in counter:
            print('{}_{}'.format(id2label[key],val),end='|')
        print()
        n_cate = len(set(Sp_label))
        acc = sum(Slabel==Sp_label)/len(Sp_label)
        accs.append(acc)
        print("top {} {} confident data acc= {}, n_rel:{},confidence:{}".format(rt,select_n,acc,n_cate,p_label_confidence[selected403[-1]]))


    class_based_index={}
    for i in range(n_rel):
        class_based_index[i]=[]
    for item in indexes[args.threshold]:
        class_based_index[train_data['p_label'][item]].append(item)

    selected403=[]
    for key,item in class_based_index.items():

        for index0,index in enumerate(item):
            if  ((index0)/len(item)<0.5):
                selected403.append(index)
        

    Slabel = np.array([train_data['label'][index] for index in selected403]) # ground-truth
    Sp_label = np.array([train_data['p_label'][index] for index in selected403]) # pseudo label
    counter=Counter()
    for item in Sp_label:
        counter[item]+=1
    counter=sorted(counter.items(),key=lambda x:x[1],reverse=True)
    print('*'*50)
    for key,val in counter:
        print('{}_{}'.format(id2label[key],val),end='|')
    print()
    n_cate = len(set(Sp_label))
    acc = sum(Slabel==Sp_label)/len(Sp_label)
    accs.append(acc)
    print("final: threshold:{} data num= {}, acc= {}, n_rel:{}".format(args.threshold,len(selected403),acc,n_cate))

    index=selected403
    selected_data = dict_index(train_data,index)

    save(selected_data,os.path.join(args.save_dir,'1.5_NL_{}_filteddouble_seed{}_threshold{}_simdouble3.pkl'.format(args.dataset,args.seed,args.threshold)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=16, help="as named")
    parser.add_argument("--dataset", type=str, default='wiki', help="as named")
    parser.add_argument("--threshold", type=float, default=0.08, help="as named")
    parser.add_argument("--save_dir", type=str, default='', help="as named")

    args=parser.parse_args()
    sampler(args)