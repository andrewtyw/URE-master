"""
用于把mnli得到的top123的结果和原始的数据合并
"""
from re import S
import sys
import os
from pathlib import Path
CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(str(PATH.parent.parent.parent.absolute()))
sys.path.append(str(PATH.parent.parent.parent.parent.absolute()))
sys.path.append(CURR_DIR)
from URE_mnli.relation_classification.utils import save,load
from URE.clean_data.clean import get_format_train_text
import numpy as np


data = load("/home/tywang/myURE/URE/TACRED/tac_six_key/dev_tac_0.01_top123.pkl")
rel2id = load("/home/tywang/myURE/URE/O2U_bert/tac_data/whole/rel2id.pkl")
data['label'] = [rel2id[item] for item in data['rel'] ]
data['top1'] = [rel2id[item] for item in data['top1']]
data['top2'] = [rel2id[item] for item in data['top2']]
data['top3'] = [rel2id[item] for item in data['top3']]
data = get_format_train_text(data)
data['pos_or_not'] = [0 if item==41 else 1 for item in data['top1']]
print(sum(np.array(data['top1'])==np.array(data['label']))/len(data['label']))
temp = load("/home/tywang/myURE/URE/O2U_bert/tac_data/whole/train_top12.pkl")
save(data,"/home/tywang/myURE/URE/O2U_bert/tac_data/whole/dev_226_acc0.8185.pkl")
#dict_keys(['text', 'rel', 'subj', 'obj', 'subj_type', 'obj_type', 'top1', 'top2', 'label', 'pos_or_not'])
# top1_p_rel = load("/home/tywang/myURE/URE_mnli/temp_files/tac_test/top1_p_rel.pkl")
# top2_p_rel = load("/home/tywang/myURE/URE_mnli/temp_files/tac_test/top2_p_rel.pkl")
# top3_p_rel = load("/home/tywang/myURE/URE_mnli/temp_files/tac_test/top3_p_rel.pkl")

# rel2id = load("/home/tywang/myURE/URE/O2U_bert/tac_data/whole/rel2id.pkl")
# tac_test_data = load("/home/tywang/myURE/URE/TACRED/tac_six_key/test.pkl")
# tac_test_data = get_format_train_text(tac_test_data)
# tac_test_data['label'] = [rel2id[item] for item in relations]
# tac_test_data['top1'] = [rel2id[item] for item in top1_p_rel]
# tac_test_data['top2'] = [rel2id[item] for item in top2_p_rel]
# tac_test_data['top3'] = [rel2id[item] for item in top3_p_rel]
# tac_test_data['pos_or_not'] = [0 if item==41 else 1 for item in tac_test_data['top1']]
# save(tac_test_data,"/home/tywang/myURE/URE/O2U_bert/tac_data/whole/test_top12.pkl")
debug_stop = 1

