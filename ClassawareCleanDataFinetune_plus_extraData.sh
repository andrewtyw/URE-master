# if you need to run extra data, make sure you've used clean data to train a TE model.
# the general input of this shell script is: 1.the detected clean data 2. TE model, which is fintuned by the clean data

## arguments
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
model_save_path=${PROJECT_PATH}"/data/model" # the 2 model will save here
MNLI_PATH=${model_save_path}"/microsoft_deberta-v2-xlarge-mnli"
BERT_PATH=${model_save_path}"/bert-base-uncased"
SEED="17" 
CUDA_INDEX="0" 
DATASET="tac"
METHOD="NL"

if [ $DATASET == "tac" ]
then
    n_rel="41"
    tags_path=${PROJECT_PATH}"/data/tac/tags.pkl"
    ratio=0.05 # \eta in the paper
    run_evaluation_path=${PROJECT_PATH}"/data/tac/test.json"
    delta=100
else
    n_rel="80"
    tags_path=${PROJECT_PATH}"/data/wiki/tags.pkl"
    ratio=0.07 # \eta in the paper
    run_evaluation_path=${PROJECT_PATH}"/data/wiki/test.pkl"
    delta=200
fi

save_info="WikifactExtraData"
clean_data_path=${PROJECT_PATH}"/data/clean_data/"${METHOD}"_sampler_"${DATASET}"_delta"${delta}"_eta"${ratio}"_SD"${SEED}".pkl"    # required 1.
extra_save_path=${PROJECT_PATH}"/data/clean_data/"${METHOD}"_sampler_"${DATASET}"_delta"${delta}"_eta"${ratio}"_SD"${SEED}"_wikifactExtra.pkl"



DICT_PATH=${PROJECT_PATH}"/data/model/"${METHOD}"_sampler_"${DATASET}"_delta"${delta}"_eta"${ratio}"_SD"${SEED}".pt" # required 2.
check_point_path=${PROJECT_PATH}"/data/model/"${METHOD}"_sampler_"${DATASET}"_delta"${delta}"_eta"${ratio}"_SD"${SEED}"_"${save_info}".pt"  # the path of the model tuned by extra data
python -u ${PROJECT_PATH}"/extra_data/wikifact_merge.py" --dataset ${DATASET} \
                                                        --clean_data_path ${clean_data_path} \
                                                        --save_path ${extra_save_path} \

echo
echo "************************************configuration************************************"
echo "seed:"${SEED}
echo "cuda_index:"${CUDA_INDEX}
echo "extra data will save here:"${extra_save_path}
echo "use pretrained model: "${DICT_PATH}
echo "finetuned model will save here:"${check_point_path}
echo "**********************START_TIME: "$(date "+%Y-%m-%d %H:%M:%S")"***********************"
echo 

# then we use the merged data to finetune TE model
python -u ${PROJECT_PATH}"/finetune/fintune_mnli.py"    --batch_size 6 \
                                --cuda_index ${CUDA_INDEX} \
                                --seed ${SEED} \
                                --lr 4e-7  \
                                --ratio ${ratio} \
                                --dataset ${DATASET} \
                                --model_path ${MNLI_PATH} \
                                --label2id_path ${PROJECT_PATH}"/data/"${DATASET}"/label2id.pkl" \
                                --selected_data_path  ${extra_save_path} \
                                --config_path  ${PROJECT_PATH}"/annotation/configs/config_"${DATASET}"_partial_constrain.json" \
                                --template2label_path ${PROJECT_PATH}"/data/"${DATASET}"/template2label.pkl" \
                                --epoch 5 \
                                --warm_up_step 300 \
                                --load_weight True \
                                --model_weight_path ${DICT_PATH} \
                                --check_point_path ${check_point_path} \

# Finally, use the fintuned NLI to infer on test set
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
                                --before_extra_dict_path ${DICT_PATH} 

