#!bin/bash

PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# mkdir 'pkl'

# path to the dbpedia
datapath=("archive/DBPEDIA_test.csv" "archive/DBPEDIA_train.csv" "archive/DBPEDIA_val.csv")

#path to save
savepath=('/data/jwwang/data/wiki-type-pretrain/pro-test.pkl' '/data/jwwang/data/wiki-type-pretrain/pro-train.pkl' '/data/jwwang/data/wiki-type-pretrain/pro-val.pkl')

for i in $(seq 0 `expr ${#datapath[@]} - 1`); do
    python data_process.py --datapath ${datapath[i]} --savepath ${savepath[i]}
    break
done

