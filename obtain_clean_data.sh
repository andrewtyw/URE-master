


######################## in progress!  #######################

# selected data will be store in finetune/selected_data 
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SEED="16"
CUDA_INDEX="3"
cd ${PROJECT_PATH}"/NLNL_bert"  # switch to run_mnli path
python -u NL.py --seed ${SEED} --epoch 5 --cuda_index ${CUDA_INDEX} --e_tags_path "/home/tywang/URE-master/data/wiki/tags.pkl"
