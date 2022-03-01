#!bin/bash

PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# mkdir 'pkl'


datapath=("/data/jwwang/data/wiki-type-pretrain/pro-train.pkl" "/data/jwwang/data/wiki-type-pretrain/pro-test.pkl" "/data/jwwang/data/wiki-type-pretrain/pro-val.pkl")

savepath=('/data/jwwang/data/wiki-type-pretrain-extend/pro-train.pkl' '/data/jwwang/data/wiki-type-pretrain-extend/pro-test.pkl' '/data/jwwang/data/wiki-type-pretrain-extend/pro-val.pkl')

for i in $(seq 0 `expr ${#datapath[@]} - 1`); do
    python data_typing.py --datapath ${datapath[i]} --savepath ${savepath[i]}
    break
done

