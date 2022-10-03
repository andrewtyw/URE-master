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
import argparse



parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str,default="microsoft_deberta-v2-xlarge-mnli", help="as named")
parser.add_argument("--cuda_index", type=int,default=3, help="as named")
parser.add_argument("--task_name", type=str,default="", help="")
parser.add_argument("--seed", type=int,default=16, help="as named")
parser.add_argument("--default_optimal_threshold", type=float,default=-1, help="as named")
parser.add_argument("--generate_data", type=bool,default=False, help="if need to generate annotation data")
parser.add_argument("--generate_data_save_path", type=str,default="./generated_data", help="folder to save generated data")
parser.add_argument("--given_save_path", type=str,default=None, help="make it to save file to specific path")



parser.add_argument("--dataset", type=str,default="wiki",choices=['tac','wiki'], help="as named")
parser.add_argument("--dict_path", type=str,default=None, help="fine-tuned model path")
parser.add_argument("--before_extra_dict_path", type=str,default=None, help="the model path before finetuned by extra data")
parser.add_argument("--load_dict", type=bool,default=False, help="if need to load the pre-trained weight,use in fewshot")
parser.add_argument("--mode", type=str,default="test", help="annotate train set or test set")
parser.add_argument("--run_evaluation_path", type=str,default="test.pkl", help="path need to evaluate")
parser.add_argument("--label2id_path", type=str,default="label2id.pkl")
parser.add_argument("--config_path", type=str,default="", help="")
parser.add_argument("--outputs", type=str,default=None, 
                help="the saved .pkl file(it would be saved in every evaluation)")
args = parser.parse_args()
print(args)


given_save_path = args.given_save_path
dataset = args.dataset
run_evaluation_path = args.run_evaluation_path
config_path = args.config_path
default_optimal_threshold = args.default_optimal_threshold
label2id_path = args.label2id_path
generate_data = args.generate_data
generate_data_save_path = args.generate_data_save_path
model_path = args.model_path
mode = args.mode
task_name = args.task_name
outputs = args.outputs
cuda_index = args.cuda_index
load_dict = args.load_dict
dict_path = args.dict_path
seed = args.seed
before_extra_dict_path = args.before_extra_dict_path

get_optimal_threshold = True if dataset=="tac" else False
current_time = TIME
out_save_path = os.path.join(CURR_DIR,"NLI_outputs/") 
