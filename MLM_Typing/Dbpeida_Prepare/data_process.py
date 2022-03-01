import pandas as pd
import pickle
import re
import argparse



def load(path:str):
    with open(path,'rb') as f:
        return pickle.load(f)

def save(path:str,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)

def filterbracket(token:str):
    return re.sub('\(.*?\)','',token)

def findSubj(text:str):
    tag=['is','was', 'are','were']
    str_centence_list = text.split('.')   
    for i in str_centence_list:
        words=i.split(' ')
        for j in tag:
            if j in words:
                return filterbracket(' '.join(words[0:words.index(j)]))
    return -1

def data_process(args):
    pd_reader = pd.read_csv(args.datapath)
        # "./archive/DBPEDIA_val.csv")

    process_data=[]

    for index,i in pd_reader.iterrows():
        subj=findSubj(i['text'])
        if subj != -1:
            process_data.append({'index':index,'text':i['text'],'subj':subj,'type':i['l2'],'type0':i['l1'],'type1': i['l3']})
        else:
            pass

    # print(filterbracket('sss (1) '))

    # print(findSubj(pd_reader.iloc[0][0]))

    # print(pd_reader.iloc[0][2])

    save(args.savepath,process_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, required=True,
                        help='dbpedia data to process')
    parser.add_argument('--savepath', type=str, required=True,
                        help='the path to save processed data')
    
    args = parser.parse_args()

    data_process(args)
    