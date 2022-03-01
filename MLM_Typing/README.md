# Steps of Typing for Entities in Wiki Dataset

## step1: prepare dbpedia Dataset
cd to MLM_Typing/Dbpedia_Prepare/ and run prepare.sh \
Then you can convert the initial data to the form which this project needs

## step2: use Flair to type the dbpedia dataset
cd to MLM_Typing/Flair_Typing and run typing.sh

## step3: train MLM_bert to type entity
cd to MLM_Typing/MLM_Training and run run.sh

## step4: use trained MLM_bert to type entities in wiki_dataset
cd to MLM_Typing/MLM_Typing and run run.sh

