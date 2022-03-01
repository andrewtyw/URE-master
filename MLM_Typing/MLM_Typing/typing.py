import glob
import os
# from flair.embeddings import TransformerWordEmbeddings
# from flair.embeddings import TransformerDocumentEmbeddings
# from flair.data import Sentence
from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np
import string
import argparse
from tqdm import tqdm
import copy
from collections import Counter
import random
import math
from os import listdir
from os.path import isfile, join, exists
import json
import copy
from torch.utils.data import DataLoader, Dataset
import pickle


def save(obj,path_name):
    with open(path_name,'wb') as file:
        pickle.dump(obj,file)

def load(path_name: object) :
    with open(path_name,'rb') as file:
        return pickle.load(file)

#自定义的dataset
class MLMTextDataset(Dataset):
    def __init__(self,args):
        data=[]

        init_data=load(args.data_path)

        for index,(text,subj,obj) in enumerate(zip(init_data['text'],init_data['subj'],init_data['obj'])):
            data.append(
            {
            'index':index,
            'subj': '[CLS] {} [SEP] {} is a [MASK] . [SEP]'.format(text,subj),
            'obj': '[CLS] {} [SEP] {} is a [MASK] . [SEP]'.format(text,obj),
            }
            )
        self.data=data

        self.type={'FootballLeagueSeason', 'Cleric', 'Politician', 'Scientist', 'Engine', 'EducationalInstitution', 'RacingDriver', 'AmusementParkAttraction', 'Athlete', 'PeriodicalLiterature', 'SportsManager', 'Eukaryote', 'Database', 'NaturalPlace', 'Software', 'RouteOfTransportation', 'SocietalEvent', 'FloweringPlant', 'WinterSportPlayer', 'WrittenWork', 'CelestialBody', 'Organisation', 'VolleyballPlayer', 'SportsTeam', 'OrganisationMember', 'Boxer', 'Horse', 'SportsEvent', 'Broadcaster', 'Cartoon', 'ClericalAdministrativeRegion', 'Actor', 'NaturalEvent', 'MotorcycleRider', 'Genre', 'Song', 'Company', 'FictionalCharacter', 'SportsTeamSeason', 'BritishRoyalty', 'Person', 'Wrestler', 'Tower', 'RaceTrack', 'Stream', 'Satellite', 'Plant', 'Olympics', 'Tournament', 'Artist', 'Coach', 'SportFacility', 'Race', 'BodyOfWater', 'Station', 'SportsLeague', 'Presenter', 'Writer', 'LegalCase', 'Comic', 'Group', 'MusicalWork', 'Animal', 'ComicsCharacter', 'MusicalArtist', 'Building', 'Venue', 'GridironFootballPlayer', 'Infrastructure', 'Settlement',
        'CARDINAL', 'WORK_OF_ART', 'DATE', 'ORDINAL', 'LANGUAGE', 'PERSON', 'TIME', 'MONEY', 'PERCENT', 'NORP', 'EVENT', 'FAC', 'QUANTITY', 'LOC', 'PRODUCT', 'ORG', 'GPE', 'LAW'}

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def test(args):
    datawithtype=[]

    top_k = 10
    stop_words = set(['than', 'to', 'for', 'about', 'same', 'by', 'where', 
            'been', 'being', 'mightn', "shan't", 'wouldn', 'me', 'us','yours', 'you' ,'he', 
            'here', "she's", 'she', 'i', "mustn't", 'y', 'our', 'those', 'haven', 'too', 
            'don', 'because', "won't", 'on', 'against', 'has', 'as', 'doing', "that'll", 
            'below', 'how', 'up', 'they', 'won', 'that', "aren't", 'some', 'so', 'theirs', 
            'didn', 'should', 'weren', 'having', 'an', 'nor', "needn't", 'of', 'yourselves', 
            'had', 'then', 'from', 'myself', 'few', "weren't", 'ours', 'couldn', 'will', 
            'needn', 'doesn', 'whom', 'themselves', "didn't", 'more', 'yourself', 'after', 
            'ain', 'are', 'does', "hasn't", 'ma', 'have', 'but', 'who', 'were', 'out', 
            'not', 'only', 'very', 'd', 'hers', 'what', 'my', 'and', "isn't", 'is', 
            'until', 'such', 'or', 't', 's', 'do', 'while', "you'll", 'it', 'their', 'am', 
            'was', 'be', 'shan', "couldn't", 'over', 'its', 'in', 'these', 've', "doesn't", 
            'we', 'can', 'hadn', 'his', "it's", 'other', 're', 'at', 'you', 'this', 'hasn', 
            'the', 'further', 'both', "you'd", 'your', "should've", 'a', 'any', 'why', 
            "shouldn't", "haven't", 'isn', 'her', "you're", 'again', "wasn't", 'did', "hadn't", 
            'own', "mightn't", 'down', 'herself', 'o', 'aren', 'shouldn', 'him', 'once', 'there', 
            'most', 'mustn', 'off', 'ourselves', 'each', 'above', 'now', 'before', 'with', 'under', 
            "don't", "wouldn't", 'which', 'if', 'when', 'himself', 'wasn', 'all', 'just', 'through', 
            'them', 'between', 'would', 'she', 'm', 'during', 'no', 'itself', "you've", 'into', 'll','.','...','?','!',
            'something','everything','nothing'])

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertForMaskedLM.from_pretrained(args.model_path)
    model.to('cuda')
    model.eval()

    dataset=MLMTextDataset(args)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.bsz)

 

    for i,batch in tqdm(enumerate(dataloader), desc="Testing"):
        tokenized_text1 = tokenizer.tokenize(batch['subj'][0])
        tokenized_text2 = tokenizer.tokenize(batch['obj'][0])

        subj_index = tokenized_text1.index('[MASK]')
        obj_index= tokenized_text2.index('[MASK]')

        tokens_index1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
        tokens_tensor1 = torch.tensor([tokens_index1]).to('cuda')

        tokens_index2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
        tokens_tensor2 = torch.tensor([tokens_index2]).to('cuda')


        predictions1 = model(tokens_tensor1)[0].detach().cpu()
        predictions2 = model(tokens_tensor2)[0].detach().cpu()



        predicted_index_subj = np.argsort(predictions1[0, subj_index])
        predicted_index_obj = np.argsort(predictions2[0, obj_index])

        predicted_subj = list(tokenizer.convert_ids_to_tokens(predicted_index_subj[-1:]))[-1]
        predicted_obj= list(tokenizer.convert_ids_to_tokens(predicted_index_obj[-1:]))[-1]

        if predicted_subj in dataset.type and predicted_obj in dataset.type:
            datawithtype.append({'index':batch['index'][0],
            'subj_type':predicted_subj,
            'obj_type':predicted_obj})
        else:
            datawithtype.append({'index':batch['index'][0],
            'subj_type':-1,
            'obj_type':-1})
    

    save(datawithtype,args.save_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--bsz', type=int, default=1,
                        help='batch size')
    parser.add_argument('--data_path', type=str, required=True,
                        help='data to type')
    parser.add_argument('--model_path', type=str, required=True,
                        help='pretrained model')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='where to save typed wiki')
    
    args = parser.parse_args()

    test(args)
    