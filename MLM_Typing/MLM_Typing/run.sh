#!bin/bash

PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python typing.py --data_path '/data/jwwang/data/wiki/wiki80_dev.pkl' \
                 --model_path '/data/jwwang/output/pretrain-wiki-entitytype-model-extend/0.967608329949533' \
                 --save_dir '/data/jwwang/output/pretrain-wiki-type-extend/dev.pkl'