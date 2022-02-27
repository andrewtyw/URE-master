


######################## wiki  #######################

# selected data will be store in finetune/selected_data 
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SEED="16"
CUDA_INDEX="1"
METHOD="O2U"  
cd ${PROJECT_PATH}"/"${METHOD}"_bert"  # switch to path


# n_rel=>num_relation (pseudo label only includes positive labels, but ground truth does have negative label)
# Equal ln_neg and n_rel would be the best setting
if [ $METHOD == "O2U" ]
then
    echo "use O2U"
    # python -u ${METHOD}.py --seed ${SEED} \   # ok
    #             --cuda_index ${CUDA_INDEX} \
    #             --dataset wiki \
    #             --e_tags_path ${PROJECT_PATH}"/data/wiki/tags.pkl" \
    #             --model_dir /data/transformers/bert-base-uncased \
    #             --train_path /home/tywang/URE-master/data/wiki/annotated/train_num40320_top1_0.4112.pkl \
    python -u ${METHOD}.py --seed ${SEED} \
                --cuda_index ${CUDA_INDEX} \
                --dataset tac \
                --e_tags_path ${PROJECT_PATH}"/data/tac/tags.pkl" \
                --model_dir /data/transformers/bert-base-uncased \
                --train_path /home/tywang/URE-master/data/tac/annotated/train_num9710_top1_0.5799.pkl \

elif [ $METHOD == "NLNL" ]
then
    echo "use NLNL"
    python -u ${METHOD}.py --seed ${SEED} \
                --epoch 5  \
                --cuda_index ${CUDA_INDEX} \
                --e_tags_path ${PROJECT_PATH}"/data/wiki/tags.pkl" \
                --n_rel 80 \
                --lr 4e-7 \
                --ln_neg 80 \
                --save_dir /home/tywang/URE-master/NLNL_bert/NLNL_out \
                --train_path /home/tywang/URE-master/data/wiki/annotated/train_num40320_top1_0.4112.pkl \

else
    echo "error"
fi
# wiki
# python -u NL.py --seed ${SEED} --epoch 5 --cuda_index ${CUDA_INDEX} --e_tags_path "/home/tywang/URE-master/data/wiki/tags.pkl"



# tac
# python -u NL.py --seed ${SEED} \
                # --epoch 10  \
                # --cuda_index ${CUDA_INDEX} \
                # --e_tags_path ${PROJECT_PATH}"/data/tac/tags.pkl" \
                # --n_rel 41 \
                # --lr 4e-7 \
                # --ln_neg 41 \
                # --save_dir /home/tywang/URE-master/NLNL_bert/NLNL_out \
#                 --train_path /home/tywang/URE-master/data/tac/annotated/train_num9710_top1_0.5799.pkl \

