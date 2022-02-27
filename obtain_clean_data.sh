


######################## in progress!  #######################

# selected data will be store in finetune/selected_data 
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SEED="16"
CUDA_INDEX="3"
cd ${PROJECT_PATH}"/NLNL_bert"  # switch to run_mnli path


# n_rel=>num_relation (pseudo label only includes positive labels, but ground truth does have negative label)
# Equal ln_neg and n_rel would be the best setting

# wiki
# python -u NL.py --seed ${SEED} --epoch 5 --cuda_index ${CUDA_INDEX} --e_tags_path "/home/tywang/URE-master/data/wiki/tags.pkl"
python -u NL.py --seed ${SEED} \
                --epoch 5  \
                --cuda_index ${CUDA_INDEX} \
                --e_tags_path ${PROJECT_PATH}"/data/wiki/tags.pkl" \
                --n_rel 80 \
                --lr 4e-7 \
                --ln_neg 80 \
                --save_dir /home/tywang/URE-master/NLNL_bert/NLNL_out \
                --train_path /home/tywang/URE-master/data/wiki/annotated/train_num40320_top1_0.4112.pkl \



# tac
# python -u NL.py --seed ${SEED} \
#                 --epoch 10  \
#                 --cuda_index ${CUDA_INDEX} \
#                 --e_tags_path ${PROJECT_PATH}"/data/tac/tags.pkl" \
#                 --n_rel 41 \
#                 --lr 4e-7 \
#                 --ln_neg 41 \
#                 --save_dir /home/tywang/URE-master/NLNL_bert/NLNL_out \
#                 --train_path /home/tywang/URE-master/data/tac/annotated/train_num9710_top1_0.5799.pkl \

