import os
import sys
os.chdir(sys.path[0])
sys.path.append('/root/tywang/URE')
sys.path.append('/root/tywang/')
import copy
import random
from tqdm import tqdm
import numpy as np
from utils.pickle_picky import *
from URE.clean_data.clean import get_format_train_text
from URE.utils.dict_relate import dict_index
from URE.clean_data.Rel2id import rel2id
random.seed(16)

def wiki_dev(wiki80_train: dict):
    """
    把一个数据集划分成80%和20%
    """
    first_key = list(wiki80_train.keys())[0]
    L = len(wiki80_train[first_key])
    selectSize = int(L*0.2)
    all_index = [i for i in range(L)]
    random.shuffle(all_index)
    selectIndex = all_index[:selectSize]
    noSelectIndex = list(set(all_index).difference(set(selectIndex)))
    dev = dict_index(wiki80_train, selectIndex)
    train = dict_index(wiki80_train, noSelectIndex)
    return train, dev


# the whole data set
# dict_keys(['text', 'rel', 'subj', 'obj', 'subj_pos', 'obj_pos']) 50400
wiki80_train = load("/root/tywang/URE/data/wiki80/wiki80_train.pkl")

# selected dataset, which has entity type
# dict_keys(['text', 'subj', 'obj', 'subj_type', 'obj_type', 'rel']) 10548   textid
train_type = load("/root/tywang/URE/data/wiki80/train_type.pkl")

# 有label的数据
# dict_keys(['text', 'textid', 'p_rel', 'subj', 'obj', 'subj_type', 'obj_type', 'subj_pos', 'obj_pos'])  1199
pretrain_data_3b_wiki = load(
    "/root/tywang/URE/data/wiki80/pretrain_data_3b_wiki.pkl")    # text id
pretrain_data_3b_wiki['rel'] = [train_type['rel'][text_id]
                                for text_id in pretrain_data_3b_wiki['textid']]  # 添加对应的relation

#dict_keys(['text', 'subj', 'obj', 'subj_type', 'obj_type', 'rel', 'text_id'])
train_10548 = load("/root/tywang/URE/data/wiki80/train_10548.pkl")
test_data = load("/root/tywang/code_for_infer_T5/data/wiki80/dev.pkl")
# 1, 使用whole data

def get_type(subj,obj,subj2type:dict,obj2type:dict):
    subj_types = []
    object_types = []
    has_type = []
    num_found = 0
    for subj,obj in zip(subj,obj):
        subj = subj.lower().strip()
        obj = obj.lower().strip()
        try:
            subj_type = subj2type[subj]
        except:
            subj_type = 'NULL_TYPE'
        try:
            obj_type = obj2type[obj]
        except:
            obj_type = 'NULL_TYPE'
        if subj_type!='NULL_TYPE' and obj_type!='NULL_TYPE':
            has_type.append(1)
            num_found+=1
        else:
            has_type.append(0)
        subj_types.append(subj_type)
        object_types.append(obj_type)
    return subj_types,object_types,has_type


def get_wiki_train_dev_test(ori_train: dict,ori_test:dict, train_type: dict, \
    pretrain_data_3b_wiki: dict,rel2id_d:dict):
    """
    ori_train 原始数据集,key包含['text', 'rel', 'subj', 'obj']
    train_type 有entity type的数据集 key包含['text', 'subj', 'obj', 'subj_type', 'obj_type', 'rel']
    pretrain_data_3b_wiki 有p_rel 的数据集 key包含['text', 'textid', 'p_rel', 'subj', 'obj', 'subj_type', 'obj_type']
    """
    train, dev = wiki_dev(ori_train)

    # 给train 注入p_rel
    pretrain_data_3b_wiki['rel'] = [train_type['rel'][text_id] for text_id in pretrain_data_3b_wiki['textid']]
    p_rels = []
    num = 0
    right = 0
    for text,subj,obj,rel in tqdm(zip(train['text'],train['subj'],train['obj'],train['rel']),total=len(train['text'])):
        p_rel = -1
        for subj1,obj1,rel1,pr in zip(pretrain_data_3b_wiki['subj'],pretrain_data_3b_wiki['obj'],pretrain_data_3b_wiki['rel'],pretrain_data_3b_wiki['p_rel']):
            if text==text and subj==subj1 and obj==obj1 and rel==rel1:

                p_rel = pr
                if rel==p_rel:
                    right+=1
                num+=1
        p_rels.append(p_rel)
    print('acc: ',right/num, ' total: ',num)
    save(p_rels,"/root/tywang/URE/data/wiki80/whole_p_rel.pkl")
    # p_rels = load("/root/tywang/URE/data/wiki80/whole_p_rel.pkl")
    train['p_rel'] = p_rels
    
    # 给train 注入entity type
    subj2type = dict(zip(train_type['subj'],train_type['subj_type']))
    obj2type = dict(zip(train_type['obj'],train_type['obj_type']))
    # subj_types = []
    # object_types = []
    # has_type = []
    # num_found = 0
    # for subj,obj in zip(train['subj'],train['obj']):
    #     subj = subj.lower().strip()
    #     obj = obj.lower().strip()
    #     try:
    #         subj_type = subj2type[subj]
    #     except:
    #         subj_type = 'NULL_TYPE'
    #     try:
    #         obj_type = obj2type[obj]
    #     except:
    #         obj_type = 'NULL_TYPE'
    #     if subj_type!='NULL_TYPE' and obj_type!='NULL_TYPE':
    #         has_type.append(1)
    #         num_found+=1
    #     else:
    #         has_type.append(0)
    #     subj_types.append(subj_type)
    #     object_types.append(obj_type)

    subj_types,object_types,has_type = get_type(train['subj'],train['obj'],subj2type,obj2type)

    train['subj_type'] = subj_types
    train['obj_type'] = object_types
    train['has_type'] = has_type

    #dev
    subj_types,object_types,has_type = get_type(dev['subj'],dev['obj'],subj2type,obj2type)

    dev['subj_type'] = subj_types
    dev['obj_type'] = object_types
    dev['has_type'] = has_type
    # test
    subj_types,object_types,has_type = get_type(ori_test['subj'],ori_test['obj'],subj2type,obj2type)

    ori_test['subj_type'] = subj_types
    ori_test['obj_type'] = object_types
    ori_test['has_type'] = has_type


    # 格式化
    train,tags1 = get_format_train_text(train,return_tag=True)
    dev,tags2 = get_format_train_text(dev,return_tag=True)
    test,tags3 = get_format_train_text(ori_test,return_tag=True)
    # rel2id
    dev['label'] = rel2id(dev['rel'],rel2id_d)
    test['label'] = rel2id(test['rel'],rel2id_d)
    train['label'] = rel2id(train['rel'],rel2id_d)
    train['p_label'] = rel2id(train['p_rel'],rel2id_d) 

    # others
    train['text1'] = train['text2'] = train['text']

    return train,dev,test

rel2id_d = load("/root/tywang/URE/data/wiki80/rel2id.pkl")
train,dev,test = get_wiki_train_dev_test(wiki80_train,test_data,train_type,pretrain_data_3b_wiki,rel2id_d)


save(train,"/root/tywang/URE/data/wiki80/train_all.pkl")
save(dev,"/root/tywang/URE/data/wiki80/dev_all.pkl")
save(test,"/root/tywang/URE/data/wiki80/test.pkl")
