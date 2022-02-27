PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SEED="16"
CUDA_INDEX="3"
cd ${PROJECT_PATH}"/URE_mnli/relation_classification"  # switch to run_mnli path
DATASET="tac"

######################## in progress!  #######################


# get optimal threshold of 0.01dev (tac)
# python -u run_evaluation.py --seed ${SEED} \
#                             --cuda_index ${CUDA_INDEX} \
#                             --dataset ${DATASET}  \
#                             --mode "0.01dev_16"  \
#                             --get_optimal_threshold True

# get train
python -u run_evaluation.py --seed ${SEED} \
                            --cuda_index ${CUDA_INDEX} \
                            --dataset ${DATASET}  \
                            --mode "train" \
                            --outputs  /home/tywang/URE-master/URE_mnli/relation_classification/mnli_out/num68124_0227010058_mnli_infer.pkl  \
                            --default_optimal_threshold 0.9379379379379379 \
                            --generate_data True
                            # --get_optimal_threshold True 
                            


# evaluate test
# get direct infer from no-finetune mnli
# python -u run_evaluation.py --seed ${SEED} --cuda_index ${CUDA_INDEX} --mode test
