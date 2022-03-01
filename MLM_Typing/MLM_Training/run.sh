#!bin/bash

PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python train.py --bert '/data/transformers/bert-large-cased' \
                --train_data_path '/data/jwwang/data/wiki-type-pretrain-extend-v2/pro-train.pkl' \
                --test_data_path '/data/jwwang/data/wiki-type-pretrain-extend-v2/pro-test.pkl' \
                --model_save_dir '/data/jwwang/output/pretrain-wiki-entitytype-model-extend-v2/'