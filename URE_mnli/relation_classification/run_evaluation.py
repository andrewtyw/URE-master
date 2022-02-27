import sys
import os
import time
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
CURR_TIME=time.strftime("%m%d%H%M%S", time.localtime())
print("current time:",CURR_TIME)
sys.path.append(CURR_DIR)
P = PATH.parent
for i in range(3): # add parent path, height = 3
    P = P.parent
    sys.path.append(str(P.absolute()))

from URE_mnli.relation_classification import arguments
from URE_mnli.relation_classification.mnli import NLIRelationClassifierWithMappingHead, REInputFeatures
from URE_mnli.relation_classification.tacred import *
# from utils.dict_relate import dict_index
from URE_mnli.relation_classification.utils import find_optimal_threshold, apply_threshold,load,save,set_global_random_seed
from URE_mnli.relation_classification.utils import find_uppercase,top_k_accuracy,dict_index
import json
from pprint import pprint
from collections import Counter
import torch

import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support
from clean_data.clean import get_format_train_text

CLASSIFIERS = {"mnli-mapping": NLIRelationClassifierWithMappingHead}
set_global_random_seed(arguments.seed)  # 设置随机种子

# tags = load("/home/tywang/myURE/URE/WIKI/typed/etags.pkl")
# dataset = load("/home/tywang/myURE/URE/WIKI/typed/wiki_trainwithtype_premnil.pkl")




    # return sum(l in p and l > 0 for l, p in zip(labels, preds)) / (labels > 0).sum()


# parser = argparse.ArgumentParser(
#     prog="run_evaluation",
#     description="Run a evaluation for each configuration.",
# )
# parser.add_argument(
#     "--input_file",
#     type=str,
#     default=arguments.run_evaluation_path,
#     help="Dataset file.",
# )
# parser.add_argument(
#     "--config",
#     type=str,
#     default=arguments.config_path,
#     help="Configuration file for the experiment.",
# )
# parser.add_argument("--basic", action="store_true", default=False)


basic=False
with open(arguments.config_path, "rt") as f:
    config = json.load(f)

# 下面的事情只有tac会干
if arguments.dataset=="tac":
    labels2id = (
        {label: i for i, label in enumerate(TACRED_LABELS)}
    )
    # id2labels
    id2labels = dict(zip(
        list(labels2id.values()),
        list(labels2id.keys())
    ))


    # with open(arguments.split_path) as file:  # tac 的split_path
    #     split_ids = file.readlines()
    #     split_ids = [item.replace("\n","") for item in split_ids]

    with open(arguments.run_evaluation_path, "rt") as f:  #输出features, labels, relations,subj_pos,obj_pos
        print("eval path:",arguments.run_evaluation_path)
        features, labels, relations,subj_pos,obj_pos = [], [],[],[],[]
        for line in json.load(f):
            id = line['id']
            # if arguments.split and arguments.selected_ratio is None:
            #     if id not in split_ids:
            #         continue
            line["relation"] = (
                line["relation"] if not basic else TACRED_BASIC_LABELS_MAPPING.get(line["relation"], line["relation"])
            )
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

    # dict_keys(['text', 'rel', 'subj', 'obj', 'subj_type', 'obj_type', 'top1', 'top2', 'label', 'pos_or_not'])
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
            cased_subj = find_uppercase(context.split(),subj.split()) #依据文本找出
            cased_obj = find_uppercase(context.split(),obj.split())
            #print(subj,"=>", cased_subj)
            # cased_subj = subj
            # cased_obj = obj
        feat_condition = str(subj_type)+':'+str(obj_type)
        if feat_condition in conditions:
            features.append(REInputFeatures(
                subj=cased_subj,
                obj=cased_obj,
                pair_type=f"{subj_type}:{obj_type}",
                context=context,
                label=label,
            ))#如果 relation_type在conditions中正常利用rules进行筛除.
        else:
            # print(feat_condition)
            features.append(REInputFeatures(
                subj=cased_subj,
                obj=cased_obj,
                pair_type=f"{subj_type}-{obj_type}",
                context=context,
                label=label,
            ))
        labels.append(labels2id[label])
        index+=1

# print(unqualified)
labels = np.array(labels)  # feature的label
print("distribution of relations",Counter(relations))


# # 根据select_ratio随机选择数据
# if arguments.selected_ratio is not None:
#     L = len(features)
#     indexes = [i for i in range(L)]
#     random.shuffle(indexes)
#     indexes = indexes[:int(L*arguments.selected_ratio)]
#     features = [features[i] for i in indexes]
#     relations = [relations[i] for i in indexes]
#     labels = labels[np.array(indexes)]




# LABEL_LIST = TACRED_BASIC_LABELS if args.basic else TACRED_LABELS

for configuration in config:
    n_labels = len(config[0]['labels'])
    _ = configuration.pop("negative_threshold", None)
    classifier = CLASSIFIERS[configuration["classification_model"]](negative_threshold=0.0, **configuration)
    output,template_socre,template_sorted, template2label = classifier(
        features,
        batch_size=configuration["batch_size"],
        multiclass=configuration["multiclass"],
    )
    if arguments.mode=="train":
        save(template2label,os.path.join(arguments.PROJECT_PATH,"data/{}/template2label.pkl".format(arguments.dataset)))
    if not "use_threshold" in configuration or configuration["use_threshold"]:
        if arguments.get_optimal_threshold:
            optimal_threshold, _ = find_optimal_threshold(labels, output)  
            print("optimal threshold:",optimal_threshold)
            # 0.01 dev optimal_threshold = 0.9379379379379379(13)  (没有finetune) 
            # 0.01 dev 0.01trai一半pos一半neg finetune  1.0
            # 0.01 dev 0.01train全pos finetune  0.997997997997998
            # 0.01 dev 0.01train全neg finetune  0.8188188188188188
            # selected by oscar 0.01 dev optimal_threshold = 0.96096
            # re-tac 0.01dev optimal_threshold 0.8758758758758759
        else:
            
            optimal_threshold = arguments.default_optimal_threshold # set default threshold
            print("use threshold:{}".format(optimal_threshold))
        
        ignore_neg_pred = False if arguments.dataset=='wiki' else True
        top1,applied_threshold_output = apply_threshold(output, threshold=optimal_threshold,ignore_negative_prediction=ignore_neg_pred)
    else:
        top1 = output.argmax(-1)

    pre, rec, f1, _ = precision_recall_fscore_support(  # 应该是只算pos的,  因为当预测全为neg_rel的时候, f1 = 0
        labels, top1, average="micro", labels=list(range(1, n_labels)) if arguments.dataset=="tac" else None
    )
    top1_acc = sum(top1==labels)/len(labels)
    top1_p_rel = [id2labels[item] for item in top1]  # get top1 relation


    configuration["precision"] = pre
    configuration["recall"] = rec
    configuration["f1-score"] = f1

    configuration["top-1"] = top1_acc
    configuration["top-2"], top2_p_rel = top_k_accuracy(applied_threshold_output, labels, k=2, id2labels=id2labels)
    configuration["top-3"], top3_p_rel = top_k_accuracy(applied_threshold_output, labels, k=3, id2labels=id2labels)
    print("labeled f1:{:.4f}".format(f1))
    print("precision:{:.4f}".format(pre))
    print("recall:{:.4f}".format(rec))
    for i in range(1,4):
        print("top{} acc={:.4f}".format(i, configuration["top-{}".format(i)]))
    

    
    
    if arguments.generate_data:
        label2id = load(arguments.label2id_path)
        id2label = dict(zip(label2id.values(),label2id.keys()))
        # save(id2label,"/home/tywang/myURE/URE_mnli/temp_files/analysis_0.01510/id2label.pkl")
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
        # save(dataset,"/home/tywang/myURE/URE/clean_data/test_data/tac_clean_testdata.pkl")

        
        dataset, etags = get_format_train_text(dataset,mode=arguments.dataset,return_tag=True)
        # current_tags = load("/home/tywang/myURE/URE/WIKI/typed/etags.pkl")
        # for tag in current_tags:
        #     if tag not in etags:
        #         etags.append(tag)
        # save(etags,"/home/tywang/myURE/URE/WIKI/typed/etags.pkl")

        # save(etags,"/home/tywang/myURE/URE/O2U_bert/tac_data/train_tags.pkl")
        # tags = load("/home/tywang/myURE/URE/O2U_bert/tac_data/train_tags.pkl")
        dataset['template'] = template_sorted
        dataset['index'] = [i for i in range(len(dataset['text']))]
        dataset['label'] = [label2id[item] for item in relations]
        dataset['top1'] = [label2id[item] for item in top1_p_rel]
        dataset['top2'] = [label2id[item] for item in top2_p_rel]
        dataset['top3'] = [label2id[item] for item in top3_p_rel]
        top1_acc = sum(np.array(dataset['label'])==np.array(dataset['top1']))/len(dataset['label'])
        _, _, f1_, _ = precision_recall_fscore_support(  # 应该是只算pos的,  因为当预测全为neg_rel的时候, f1 = 0
        dataset['label'], dataset['top1'] , average="micro", labels=list(range(1, n_labels)) if arguments.dataset=="tac" else None
        )
        print("*"*10,"selected data information","*"*10)
        print("top1 acc: ",top1_acc)
        print("labeled f1: ",f1_)
        print("*"*30)
        print("start to generate inferred data...")
        if arguments.dataset=="tac":
            # for tac, we just select datas with positive pseudo label
            neg_id = label2id['no_relation']
            pos_index = [i for i in range(len(dataset['top1'])) if dataset['top1'][i]!=neg_id]
            dataset = dict_index(dataset,pos_index)
            top1_acc = sum(np.array(dataset['top1'])==np.array(dataset['label']))/len(dataset['top1'])
            print("tac selected acc:{:.4f}".format(top1_acc))
        save_path = os.path.join(arguments.generate_data_save_path,"{}_{}.pkl".format(arguments.dataset,arguments.mode))
        save(dataset,save_path)