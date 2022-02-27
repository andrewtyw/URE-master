import sys
import os
import time
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())  # '/home/tywang/URE-master/URE_mnli/relation_classification'
CURR_TIME=time.strftime("%m%d%H%M%S", time.localtime())
sys.path.append(CURR_DIR)
P = PATH.parent
PROJECT_PATH  = "" 
for i in range(3): # add parent path, height = 3
    P = P.parent
    if i==1: PROJECT_PATH = str(P.absolute())
    sys.path.append(str(P.absolute()))
import argparse





parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str,default="/data/transformers/microsoft_deberta-v2-xlarge-mnli", help="as named")
parser.add_argument("--cuda_index", type=int,default=3, help="as named")
parser.add_argument("--task_name", type=str,default="mnli_infer", help="as named")
parser.add_argument("--dict_path", type=str,default=None, help="fine-tuned model path")
# wiki "/data/tywang/O2U_model/fine_mnli_wiki_NLNL_num4032_epo0-0.9509.pt"
parser.add_argument("--load_dict", type=bool,default=False, help="as named")
parser.add_argument("--dataset", type=str,default="wiki",choices=['tac','wiki'], help="as named")
parser.add_argument("--seed", type=int,default=16, help="as named")
parser.add_argument("--mode", type=str, help="as named")
parser.add_argument("--generate_data", type=bool,default=False, help="as named")
parser.add_argument("--get_optimal_threshold", type=bool,default=False, help="if it is True, the following default_optimal_threshold would not be used")
parser.add_argument("--default_optimal_threshold", type=float,default=0.9379379379379379, help="optimal threshold in 0.01dev in tac")
parser.add_argument("--out_save_path", type=str,default=None, 
                help="the path is used to save the original output, which is a matrix with shape=[n_data, n_relation]. \
                 If None, the matrix will be calculated, else, you need to specify the path then it would be loaded(that saves time)")
parser.add_argument("--outputs", type=str,default=None, 
                help="the output matrix mentioned above")
args = parser.parse_args()
print(args)

model_path = args.model_path
cuda_index = args.cuda_index
task_name = args.task_name
dict_path = args.dict_path
load_dict = args.load_dict
dataset = args.dataset
mode = args.mode
seed = args.seed
outputs = args.outputs  # if output has been computed, please specify it's path in args.out_save_path
get_optimal_threshold = args.get_optimal_threshold
generate_data = args.generate_data  # generate data with pseudo label



# """调试模式(把sh文件中的参数设定到这里来)"""
# dataset = "tac"
# mode = "train"
# outputs = "/home/tywang/URE-master/URE_mnli/relation_classification/mnli_out/num68124_0227010058_mnli_infer.pkl"
# get_optimal_threshold = False
# default_optimal_threshold = 0.9129129129129129
"""调试模式(把sh文件中的参数设定到这里来)"""


out_save_path = os.path.join(CURR_DIR,"mnli_out")
generate_data_save_path = os.path.join(PROJECT_PATH,"data/"+dataset+"/annotated/")
config_path = os.path.join(CURR_DIR,"configs/config_{}_partial_constrain.json".format(dataset))
label2id_path = os.path.join(PROJECT_PATH,"data/"+dataset+"/label2id.pkl")
if dict_path==None: load_dict=False




# choose the data needed to infer p_label
if dataset=="tac":
    default_optimal_threshold = args.default_optimal_threshold
    run_evaluation_path = os.path.join(PROJECT_PATH,"data/"+dataset+"/raw/{}.json".format(mode))
elif dataset=="wiki":
    get_optimal_threshold = False
    default_optimal_threshold = 0
    run_evaluation_path = os.path.join(PROJECT_PATH,"data/"+dataset+"/typed/wiki_{}withtype_premnil.pkl".format(mode))
if not os.path.exists(run_evaluation_path):
    print("data {} not found !".format(run_evaluation_path))
    sys.exit()




# # 公用
# #  "/data/transformers/microsoft_deberta-v2-xlarge-mnli"
# #  "/data/transformers/microsoft_deberta-v2-xxlarge-mnli"  
# model_path = "/data/transformers/microsoft_deberta-v2-xlarge-mnli"  #
# current_time=time.strftime("%m%d%H%M%S", time.localtime())# 记录被初始化的时间  #
# print(current_time)
# cuda_index = 3 #
# seed = 16
# task_name = "wiki_fewshot_generate_p_label"
# dict_path = "/data/tywang/O2U_model/fine_mnli_wiki_NLNL_num4032_epo0-0.9509.pt" # "/data/tywang/O2U_model/fine_mnli_only_neg_Label_NEW_NEG_TEMPLATE_acc_v1-0.9779.pt" # /data/tywang/O2U_model/fine_mnli_only_pos_acc_v1-0.9926.pt 
# load_dict = False                                                         # /data/tywang/O2U_model/fine_mnli_only_neg_acc_v1-0.9601.pt
#                                                                          # /data/tywang/O2U_model/fine_mnli_acc_v1-0.9767.pt
# ################### bool  ##################




# """
# tac 专用
# """

# dataset = "tac"
# out_save_path = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out"

# config_path = "/home/tywang/myURE/URE_mnli/relation_classification/configs/config_tac_partial_constrain.json"  # run evaluation 的config path
# label2id_path = "/home/tywang/myURE/URE/O2U_bert/tac_data/whole/rel2id.pkl"


# outputs = None # 需要计算mnli的时候, 在跑dev set的时候要把output设置为None
# outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num68124_02-11-203242_WholeProc0.1.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-12-184928_WholeProc0.05_NEUTRAL.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-10-221227_WholeProc0.1.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-10-160724_tac_test_seed13_no_finetune.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-09-142635_0.01train.pkl"
# # outputs ="/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/time_12-10-012219_model_-v2-xlarge-mnli_nFea_68124_output.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/tac_xlarge_dev_22631.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/tac_xlarge_train_68124.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-10-151430_tac_test_seed13_no_finetune.pkl"  # tac test ori
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num13418_01-14-033752_retac_test.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-06-003328_tac_test_ok_v1_.pkl"  # 一半pos一半neg finetune
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-06-111951_tac_test_ok_v1_.pkl"  # 全pos
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-06-133146_tac_test_ok_v1_.pkl"  # 全neg

# mode = "train"
# tac_data_path = "/home/tywang/myURE/URE/TACRED/tacred_norel"  # 名下有 train/dev/test.json
# run_evaluation_path = os.path.join(tac_data_path,"{}.json".format(mode))  # tac test
# # run_evaluation_path = "/home/tywang/myURE/URE/Re-TACRED/reTAC-output/test.json"  # retac test


# split_path = "/home/tywang/myURE/Ask2Transformers_old_version/resources/tacred_splits/train/0.01.split.txt"   # "/home/tywang/myURE/Ask2Transformers/resources/tacred_splits/dev/dev.0.01.split.txt"  # oscar作者本人选出的0.01的split
# split = False  # 在数据中(eg dev.json) 随机选择一定比例的数据来跑 加载作者的数据
# generate_data = True
# get_optimal_threshold = False # 是否计算该数据集对应的optimal threshold
# default_optimal_threshold = 0.9379379379379379
# save_dataset_name = None # "test.pkl"  如果不save就写None

# selected_ratio =None #0.01 # 不select就写None  随机自选


# """
# wiki 专用
# """
# # dataset = "wiki"
# # out_save_path = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_wiki_out"
# # config_path = "/home/tywang/myURE/URE_mnli/relation_classification/configs/config_wiki_partial_constraint.json"
# # label2id_path = "/home/tywang/myURE/URE/WIKI/typed/label2id.pkl"
# # outputs = None
# # # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_wiki_out/num40320_0223171220_wiki_fewshot_generate_p_label.pkl"  # 需要自己跑
# # generate_data = False
# # mode = "test"
# # wiki_data_path = "/home/tywang/myURE/URE/WIKI/typed"  # 名下有 train/dev/test 的数据
# # run_evaluation_path = os.path.join(wiki_data_path,"wiki_{}withtype_premnil.pkl".format(mode))  # tac test
# # get_optimal_threshold = False
# # default_optimal_threshold = 0
# # selected_ratio =None 

# # """
# # cd /home/tywang/myURE/URE_mnli/relation_classification
# # nohup python -u run_evaluation.py >/home/tywang/myURE/URE_mnli/relation_classification/logs/xx.log 2>&1 &
# # """