import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(CURR_DIR)
P = PATH.parent
for i in range(3): 
    P = P.parent
    sys.path.append(str(P.absolute()))
import time
TIME=time.strftime("%m-%d-%H*%M*%S", time.localtime())
print("time",TIME)

import argparse
import json
from collections import Counter

import numpy as np
import random
import torch 
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from mnli import NLIRelationClassifierWithMappingHead, REInputFeatures
from tacred import *
import arguments
from find_threshold import get_optimal_threshold
from annotation_utils import find_optimal_threshold, apply_threshold,load,save,set_global_random_seed
from annotation_utils import find_uppercase,top_k_accuracy,dict_index
from utils.clean import get_format_train_text
set_global_random_seed(arguments.seed)  
import warnings
warnings.filterwarnings("ignore")

if arguments.dataset=="wiki":
    wiki_data = load(arguments.run_evaluation_path)




with open(arguments.config_path, "rt") as f:
    config = json.load(f)




"""
    to obtain:
    features: List[REInputFeatures],
    labels: List[int],
    relations: List[str],
    subj_pos: List[List]
    obj_pos: List[List]
"""
if arguments.dataset=="tac":
    labels2id = (
        {label: i for i, label in enumerate(TACRED_LABELS)}
    )
    
    id2labels = dict(zip(
        list(labels2id.values()),
        list(labels2id.keys())
    ))

    
    with open(arguments.run_evaluation_path, "rt") as f:  
        features, labels, relations,subj_pos,obj_pos = [], [],[],[],[]
        for line in json.load(f):
            id = line['id']
            subj_posistion= [line["subj_start"] , line["subj_end"]]
            subj_pos.append(subj_posistion)
            obj_posistion= [line["obj_start"] , line["obj_end"]]
            obj_pos.append(obj_posistion)
            features.append(
                REInputFeatures(
                    subj=" ".join(line["token"][line["subj_start"] : line["subj_end"] + 1])
                    .replace("-LRB-", "(")
                    .replace("-RRB-", ")")
                    .replace("-LSB-", "[")
                    .replace("-RSB-", "]"),
                    obj=" ".join(line["token"][line["obj_start"] : line["obj_end"] + 1])
                    .replace("-LRB-", "(")
                    .replace("-RRB-", ")")
                    .replace("-LSB-", "[")
                    .replace("-RSB-", "]"),
                    pair_type=f"{line['subj_type']}:{line['obj_type']}",
                    context=" ".join(line["token"])
                    .replace("-LRB-", "(")
                    .replace("-RRB-", ")")
                    .replace("-LSB-", "[")
                    .replace("-RSB-", "]"),
                    label=line["relation"],
                )
            )
            relations.append(line["relation"])
            labels.append(labels2id[line["relation"]])

elif arguments.dataset=="wiki":
    wiki_labels = config[0]['labels']
    labels2id = dict(zip(wiki_labels,[i for i in range(len(wiki_labels))]))
    id2labels = dict(zip(labels2id.values(),labels2id.keys()))
    wiki_data = load(arguments.run_evaluation_path)
    try: 
        del wiki_data['index']
    except:
        pass
    features, labels = [], []
    relations = wiki_data['rel']
    subj_pos = wiki_data['subj_pos']
    obj_pos = wiki_data['obj_pos']

    conditions=[]
    for i,j in config[0]['valid_conditions'].items():
        conditions.extend(j)
    index = 0
    unqualified = 0
    for context, subj, obj, subj_type,obj_type,subj_p,obj_p,label in zip(
       wiki_data['text'], wiki_data['subj'],wiki_data['obj'],
        wiki_data['subj_type'], wiki_data['obj_type'],
        wiki_data['subj_pos'], wiki_data['obj_pos'],
        wiki_data['rel'],
    ):
        cased_subj = ' '.join(context.split()[subj_p[0]:subj_p[1]])
        cased_obj = ' '.join(context.split()[obj_p[0]:obj_p[1]])
        try:
            
            assert subj.replace(" ","")==cased_subj.lower().replace(" ","")
            assert obj.replace(" ","")==cased_obj.lower().replace(" ","")
        except:
            
            unqualified+=1 
            cased_subj = find_uppercase(context.split(),subj.split()) 
            cased_obj = find_uppercase(context.split(),obj.split())
            
            
            
        feat_condition = str(subj_type)+':'+str(obj_type) 
        if feat_condition in conditions:  
            features.append(REInputFeatures(
                subj=cased_subj,
                obj=cased_obj,
                pair_type=f"{subj_type}:{obj_type}",
                context=context,
                label=label,
            ))
        else:
            
            features.append(REInputFeatures(
                subj=cased_subj,
                obj=cased_obj,
                pair_type=f"{subj_type}?{obj_type}",
                context=context,
                label=label,
            ))
        labels.append(labels2id[label])
        index+=1


labels = np.array(labels)  






for configuration in config:
    n_labels = len(config[0]['labels'])
    _ = configuration.pop("negative_threshold", None)
    classifier = NLIRelationClassifierWithMappingHead(negative_threshold=0.0, **configuration)
    output,entailment_score,template_sorted, template2label = classifier(
        features,
        batch_size=configuration["batch_size"],
        multiclass=configuration["multiclass"],
    )




    if arguments.mode=="dev" and arguments.dataset=="tac":
        print("dev, automatically find threshold")
        optimal_threshold, _ = find_optimal_threshold(labels, output)  
    elif arguments.dataset=="tac":
        if arguments.default_optimal_threshold>0:
            print("set default threshold")
            optimal_threshold = arguments.default_optimal_threshold
        else:
            optimal_threshold = get_optimal_threshold()
    else:
        
        optimal_threshold = 0

    ignore_neg_pred = False if arguments.dataset in ['wiki','wikifact'] else True
    print("use threshold:",optimal_threshold)
    top1,applied_threshold_output,score = apply_threshold(output, threshold=optimal_threshold,ignore_negative_prediction=ignore_neg_pred)


    
    label_scores = []
    for entail_score in entailment_score:
        label_score = {}
        for label,e_score in enumerate(entail_score):
            label_score[id2labels[label]] = e_score
        label_scores.append(label_score)
    top1_rels = [id2labels[id] for id in top1]

    score = F.normalize(torch.from_numpy(applied_threshold_output),p=1,dim=1)
    score = score.max(-1)[0].tolist()

    

    pre, rec, f1, _ = precision_recall_fscore_support(  
        labels, top1, average="micro", labels=list(range(1, n_labels)) if arguments.dataset=="tac" else None
    )
    macro_pre, macro_rec, macro_f1, _ = precision_recall_fscore_support(  
        labels, top1, average="macro", labels=list(range(1, n_labels)) if arguments.dataset=="tac" else None
    )
    
    macro_pre_nolabel, macro_rec_nolabel, macro_f1_nolabel, _ = precision_recall_fscore_support(  
        labels, top1, average="macro", labels= None
    )
    micro_pre_nolabel, micro_rec_nolabel, micro_f1_nolabel, _ = precision_recall_fscore_support(  
        labels, top1, average="micro", labels= None
    )
    top1_acc = sum(top1==labels)/len(labels)
    top1_p_rel = [id2labels[item] for item in top1]  


    configuration["precision"] = pre
    configuration["recall"] = rec
    configuration["f1-score"] = f1

    configuration["top-1"] = top1_acc
    configuration["top-2"], top2_p_rel = top_k_accuracy(applied_threshold_output, labels, k=2, id2labels=id2labels)
    configuration["top-3"], top3_p_rel = top_k_accuracy(applied_threshold_output, labels, k=3, id2labels=id2labels)
    print("*"*20,"Micro result","*"*20)
    print("f1(micro):{:.6f}".format(f1))
    print("precision(micro):{:.6f}".format(pre))
    print("recall(micro):{:.6f}".format(rec))
    print("*"*20,"Macro result","*"*20)
    print("f1(macro):{:.6f}".format(macro_f1))
    print("precision(macro):{:.6f}".format(macro_pre))
    print("recall(macro):{:.6f}".format(macro_rec))
    
    
    print("*"*20,"Top K accuracy","*"*20)
    for i in range(1,4):
        print("top{} acc={:.6f}".format(i, configuration["top-{}".format(i)]))
    
    
    
    if arguments.generate_data:
        label2id = load(arguments.label2id_path)
        id2label = dict(zip(label2id.values(),label2id.keys()))
        
        dataset = {
        'text':[],
        'rel':[],
        'subj':[],
        'obj':[],
        'subj_pos':subj_pos,
        'obj_pos':obj_pos,
        'subj_type':[],
        'obj_type':[],
        }
        assert len(features)==len(relations)
        for feat,rel in zip(features,relations):
            feat:REInputFeatures
            dataset['text'].append(feat.context)
            dataset['rel'].append(rel)
            dataset['subj'].append(feat.subj)
            dataset['obj'].append(feat.obj)
            
            if ":" in feat.pair_type:
                subj_type,obj_type = feat.pair_type.split(":")
            elif "-" in feat.pair_type: 
                subj_type,obj_type = feat.pair_type.split("-")
            dataset['subj_type'].append(subj_type)
            dataset['obj_type'].append(obj_type)
        if arguments.dataset=="tac":
            for text,subj, subj_p, obj,obj_p in zip(dataset['text'],dataset['subj'],dataset['subj_pos'],dataset['obj'],dataset['obj_pos']):
                assert ' '.join(text.split()[subj_p[0]:subj_p[1]+1])==subj
                assert ' '.join(text.split()[obj_p[0]:obj_p[1]+1])==obj

        
        dataset, etags = get_format_train_text(dataset,mode=arguments.dataset,return_tag=True)

        dataset['template'] = template_sorted
        dataset['score'] = score
        dataset['index'] = [i for i in range(len(dataset['text']))]
        dataset['label'] = [label2id[item] for item in relations]
        dataset['top1'] = [label2id[item] for item in top1_p_rel]
        dataset['top2'] = [label2id[item] for item in top2_p_rel]
        dataset['top3'] = [label2id[item] for item in top3_p_rel]
        top1_acc = sum(np.array(dataset['label'])==np.array(dataset['top1']))/len(dataset['label'])
        _, _, f1_, _ = precision_recall_fscore_support( 
        dataset['label'], dataset['top1'] , average="micro", labels=list(range(1, n_labels)) if arguments.dataset=="tac" else None
        )
        print("*"*10,"selected data information","*"*10)
        print("top1 acc: ",top1_acc)
        print("labeled f1: ",f1_)
        print("*"*30)
        print("start to generate inferred data...")
        if arguments.dataset=="tac" and arguments.mode=="train":
            neg_id = label2id['no_relation']
            pos_index = [i for i in range(len(dataset['top1'])) if dataset['top1'][i]!=neg_id]  
            
            dataset_pos = dict_index(dataset,pos_index)
            top1_acc = sum(np.array(dataset_pos['top1'])==np.array(dataset_pos['label']))/len(dataset_pos['top1'])
            print("tac selected acc:{:.6f}".format(top1_acc))
            print("tac selected num:{}".format(len(dataset_pos['top1'])))


            if arguments.given_save_path is not None:
                save(dataset_pos,arguments.given_save_path )
            else:  
                save_path = os.path.join(arguments.generate_data_save_path,"{}_{}_pos_T{}_acc{:.4f}.pkl".format(arguments.dataset,arguments.mode,TIME,top1_acc))
                save(dataset_pos,save_path)
            
            
            
            

        else:
            if arguments.given_save_path is not None:
                save(dataset,arguments.given_save_path )
            
            else:
                save_path = os.path.join(arguments.generate_data_save_path,"{}_{}_T{}_acc{:.4f}.pkl".format(arguments.dataset,arguments.mode,TIME,top1_acc))
                save(dataset,save_path)
