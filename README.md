# URE-master
## Environment
In order to exactly reproduce the result, please setup the environment by `requirements.txt`
## Dataset
`TACRED` is not provided in this repository, but you could obtain it from [here](https://nlp.stanford.edu/projects/tacred/). After downloading please put `{train,dev,test}.json` into `/URE-master/data/tac`.
## Run And Reproduce
1. Clean Data Finetune
```
sh CleanDataFinetune.sh
```
2. Class-aware Clean Data Finetune (Dynamic)
```
sh ClassawareCleanDataFinetune.sh
```
---
3. Clean Data Finetune + extra data
```
sh CleanDataFinetune_plus_extraData.sh
```
4. Class-aware Clean Data Finetune (Dynamic) + extra data
```
sh ClassawareCleanDataFinetune_plus_extraData.sh
```
**We provided the logs for each `.sh` file in /logs_sample**

