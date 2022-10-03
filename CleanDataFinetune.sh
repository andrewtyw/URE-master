PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SEED="16" 
CUDA_INDEX="1"  
DATASET="tac"  # choice in ["tac","wiki"], stands for TACRED and Wiki80
specified_save_path=${PROJECT_PATH}"/data/clean_data"  # the selected clean data will store here
METHOD="NL"    # choice in ["NL","O2U","DivideMix"]   the method to obtain clean data
save_info=$(date "+%m%d%H%M%S")  # marker

# if you have the following two huggingface models, specifiy their paths
model_save_path=${PROJECT_PATH}"/data/model" # the 2 model will save here
MNLI_PATH=${model_save_path}"/microsoft_deberta-v2-xlarge-mnli"
BERT_PATH=${model_save_path}"/bert-base-uncased"

# 
if [ $DATASET == "tac" ]
then
    n_rel="41"  # we select annotated data with positive pseudo label, hence n_rel=41
    template2label_path=${PROJECT_PATH}"/data/tac/template2label.pkl"
    label2id=${PROJECT_PATH}"/data/tac/label2id.pkl"
    ratio=0.05 # select 0.05*Ntrain clean data  (\eta in the paper)
    run_evaluation_path=${PROJECT_PATH}"/data/tac/test.json"  # rac test set
    train_path=${PROJECT_PATH}"/data/tac/train.json"
    delta=100
else
    n_rel="80"
    template2label_path=${PROJECT_PATH}"/data/wiki/template2label.pkl"
    label2id=${PROJECT_PATH}"/data/wiki/label2id.pkl"
    ratio=0.07 # select 0.07*Ntrain clean data
    run_evaluation_path=${PROJECT_PATH}"/data/wiki/test.pkl" # evaluate on wiki80 test set
    train_path=${PROJECT_PATH}"/data/wiki/train.pkl"
    delta=200
fi

config_path=${PROJECT_PATH}"/annotation/configs/config_"${DATASET}"_partial_constrain.json"
annotated_path=${PROJECT_PATH}"/data/annotation_result/"${DATASET}"_annotation.pkl"  # annotated data will save here
clean_data_path=${specified_save_path}"/"${METHOD}"_"${DATASET}"_RT"${ratio}"_SD"${SEED}".pkl"  
check_point_path=${PROJECT_PATH}"/data/model/"${METHOD}"_"${DATASET}"_RT"${ratio}"_SD"${SEED}".pt" 



echo
echo "************************************configuration************************************"
echo "seed:"${SEED}
echo "cuda_index:"${CUDA_INDEX}
echo "noisy_data_path:"${annotated_path}
echo "n_relation:"${n_rel}
echo "clean_data_path:"${clean_data_path}
echo "fine_NLI_model_path:"${check_point_path}
echo "**********************START_TIME: "$(date "+%Y-%m-%d %H:%M:%S")"***********************"
echo 

# unzip dataset
cd ${PROJECT_PATH}/data/tac
unzip -o tac.zip
cd ${PROJECT_PATH}/data/wiki
unzip -o wiki.zip

# download huggingface models
# if you have already had 'microsoft_deberta-v2-xlarge-mnli' and 'bert-base-uncased', you can skip this
python -u ${PROJECT_PATH}/utils/prepare.py --model_save_folder ${model_save_path}


# Stage 1: annotate train to get silver data
python -u ${PROJECT_PATH}/annotation/run_evaluation.py \
                                --model_path ${MNLI_PATH} \
                                --cuda_index ${CUDA_INDEX} \
                                --dataset ${DATASET} \
                                --seed ${SEED} \
                                --mode "train" \
                                --run_evaluation_path ${train_path} \
                                --label2id_path ${PROJECT_PATH}"/data/"${DATASET}"/label2id.pkl" \
                                --config_path ${config_path} \
                                --given_save_path ${annotated_path} \
                                --generate_data True \





# Stage 2: get clean data
python -u ${PROJECT_PATH}"/cleaning/NL.py" \
                                    --seed ${SEED} \
                                    --epoch 10  \
                                    --cuda_index ${CUDA_INDEX} \
                                    --e_tags_path ${PROJECT_PATH}"/data/"${DATASET}"/tags.pkl" \
                                    --n_rel ${n_rel} \
                                    --lr 4e-7 \
                                    --ln_neg ${n_rel} \
                                    --train_path ${annotated_path} \
                                    --specified_save_path ${specified_save_path} \
                                    --eta ${ratio} \
                                    --delta ${delta} \
                                    --dataset ${DATASET} \
                                    --label2id ${label2id}



# # Stage 3: use clean data to finetune NLI
python -u ${PROJECT_PATH}"/finetune/fintune_mnli.py"    --batch_size 6 \
                                --cuda_index ${CUDA_INDEX} \
                                --seed ${SEED} \
                                --lr 4e-7  \
                                --ratio ${ratio} \
                                --dataset ${DATASET} \
                                --model_path ${MNLI_PATH} \
                                --label2id_path ${PROJECT_PATH}"/data/"${DATASET}"/label2id.pkl" \
                                --selected_data_path  ${clean_data_path} \
                                --config_path  ${PROJECT_PATH}"/annotation/configs/config_"${DATASET}"_partial_constrain.json" \
                                --template2label_path ${PROJECT_PATH}"/data/"${DATASET}"/template2label.pkl" \
                                --epoch 10 \
                                --check_point_path ${check_point_path} \

# # Finally, use the fintuned NLI to infer on test set
python -u ${PROJECT_PATH}"/annotation/run_evaluation.py" \
                                --model_path ${MNLI_PATH} \
                                --cuda_index ${CUDA_INDEX} \
                                --dataset ${DATASET} \
                                --seed ${SEED} \
                                --mode "test" \
                                --run_evaluation_path ${run_evaluation_path} \
                                --label2id_path ${PROJECT_PATH}"/data/"${DATASET}"/label2id.pkl" \
                                --config_path ${PROJECT_PATH}"/annotation/configs/config_"${DATASET}"_partial_constrain.json" \
                                --load_dict True \
                                --dict_path ${check_point_path} \
