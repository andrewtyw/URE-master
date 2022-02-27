echo $(date)
# 1, type wiki

# 2, prepare pseudo label
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SEED="16"
CUDA_INDEX="1"
METHOD="NLNL" 
DATASET="wiki"  # choice in ["tac","wiki"]
RATIO=0.05  # choice in [0.01,0.05,0.1]
MNLI_PATH="/data/transformers/microsoft_deberta-v2-xlarge-mnli"
BERT_PATH="/data/transformers/bert-base-uncased"

cd ${PROJECT_PATH}"/URE_mnli/relation_classification"  # switch to run_mnli path


python -u run_evaluation.py --seed ${SEED} \
                            --cuda_index ${CUDA_INDEX} \
                            --dataset ${DATASET}  \
                            --mode "train" \
                            --default_optimal_threshold 0 \

# 3, select high confident pseudo label
cd ${PROJECT_PATH}"/"${METHOD}"_bert"
if [ $METHOD == "O2U" ]
then
    echo "use O2U"
    python -u ${METHOD}.py --seed ${SEED} \
                --cuda_index ${CUDA_INDEX} \
                --dataset wiki \
                --batch_size 64 \
                --e_tags_path ${PROJECT_PATH}"/data/wiki/tags.pkl" \
                --model_dir /data/transformers/bert-base-uncased \
                --train_path ${PROJECT_PATH}"/data/wiki/annotated/wiki_train.pkl" \

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
                --train_path ${PROJECT_PATH}"/data/wiki/annotated/wiki_train.pkl" \

else
    echo "error"
fi

# 4, use high confident pseudo label finetune MNLI
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
                                --model_path ${MNLI_PATH} \
                                # --load_weight "False" \
                                # --model_weight_path  "/data/tywang/O2U_model/fine_mnli_wiki_random_fewshot_0.05_num50_epo3-0.9333.pt"


# 5 use the finetune date infer test set
python -u run_evaluation.py --seed ${SEED} \
                            --cuda_index ${CUDA_INDEX} \
                            --dataset ${DATASET}  \
                            --mode "test" \
                            --default_optimal_threshold 0 \
                            --load_dict True  \
                            --dict_path ${PROJECT_PATH}"/data/save_model/wiki_n"${RATIO}"train.pt"

echo $(date)