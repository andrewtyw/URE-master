# if you need to run extra data, make sure you've used clean data to train a TE model.
# the general input of this shell script is: 1.the detected clean data 2. TE model, which is fintuned by the clean data

## arguments
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
model_save_path=${PROJECT_PATH}"/data/model" # the 2 model will save here
MNLI_PATH=${model_save_path}"/microsoft_deberta-v2-xlarge-mnli"
BERT_PATH=${model_save_path}"/bert-base-uncased"
SEED="16" 
CUDA_INDEX="2" 
DATASET="wiki"

if [ $DATASET == "tac" ]
then
    n_rel="41"
    tags_path=${model_save_path}"/data/tac/tags.pkl"
    ratio=0.05
    run_evaluation_path=${model_save_path}"/data/tac/test.json"
else
    n_rel="80"
    tags_path=${model_save_path}"/data/wiki/tags.pkl"
    ratio=0.07
    run_evaluation_path=${model_save_path}"/data/wiki/test.pkl"
fi

save_info="WikifactExtraData"
INFO="wikifact4"${DATASET}"_"${SEED}
specified_save_path=${PROJECT_PATH}"/extra_data/outputs/"${INFO}".pkl"
clean_data_path=${PROJECT_PATH}"/data/clean_data/NL_"${DATASET}"_RT"${ratio}"_SD"${SEED}".pkl"  # required 1.
extra_save_path=${PROJECT_PATH}"/data/clean_data/NL_"${DATASET}"_RT"${ratio}"_SD"${SEED}"_wikifactExtra.pkl"



DICT_PATH=${PROJECT_PATH}"/data/model/NL_"${DATASET}"_RT"${ratio}"_SD"${SEED}".pt"  # required 2.
check_point_path=${PROJECT_PATH}"/data/model/"${METHOD}"_"${DATASET}"_RT"${ratio}"_SD"${SEED}"_"${save_info}".pt"  # fine-tune好的NLI的路径 将会是这个
python -u ${PROJECT_PATH}"/extra_data/wikifact_merge.py" --dataset ${DATASET} \
                                                        --clean_data_path ${clean_data_path} \
                                                        --save_path ${extra_save_path} \


# then we use the merged data to finetune TE model
cd ${PROJECT_PATH}"/finetune" 
python -u fintune_mnli.py    --batch_size 6 \
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
                                --default_optimal_threshold 1.0 \
                                # Note that, when using extra data on TACRED, the threshold should be the threshold before using extraData, 
                                # i.e. the threshold of after using Clean data fintune. Here set it to 1.0 as a placeholder.
