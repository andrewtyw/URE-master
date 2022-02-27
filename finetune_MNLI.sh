
######################## in progress!  #######################

PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SEED="16"
CUDA_INDEX="2"
RATIO=0.05  # 
echo ${WARMIP_STEPS}
DATASET="tac"  # you need to choose a dataset, choice in ["tac","wiki"]
cd ${PROJECT_PATH}"/finetune" 
# Convert pseudo data into data for training MNLI (3 categories: entailment, contraction, neutral)
python -u tacred2mnli.py    --label2id_path ${PROJECT_PATH}"/data/"${DATASET}"/label2id.pkl" \
                            --template2label_path ${PROJECT_PATH}"/data/"${DATASET}"/template2label.pkl" \
                            --selected_data_path  ${PROJECT_PATH}"/finetune/selected_data/"${DATASET}"/selected_n"${RATIO}"train.pkl" \
                            --config_path  ${PROJECT_PATH}"/URE_mnli/relation_classification/configs/config_"${DATASET}"_partial_constrain.json" \
                            --ratio ${RATIO}
# use the above generated data to Finetune
python -u fintune_mnli_v3.py    --batch_size 6 \
                                --cuda_index ${CUDA_INDEX} \
                                --seed ${SEED} \
                                --ratio ${RATIO} \
                                --dataset ${DATASET} \
                                --train_path ${PROJECT_PATH}"/finetune/finetune_data/"${DATASET}"/finetune_n"${RATIO}"train.pkl" \
                                --model_path "/data/transformers/microsoft_deberta-v2-xlarge-mnli" \
                                --load_weight "False" \
                                --model_weight_path  "/data/tywang/O2U_model/fine_mnli_wiki_random_fewshot_0.05_num50_epo3-0.9333.pt"
