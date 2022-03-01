import argparse
from json import load
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertPreTrainedModel
from transformers import AlbertModel, AlbertPreTrainedModel
from transformers import BertTokenizer, BertForMaskedLM

import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset

import pickle

from tqdm import tqdm
import wandb
import numpy as np

# default_config = dict(
#     epoch=1,
#     batch_size=5,
#     device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
#     test_batch_size=20,
# )

# wandb.init(config=default_config)


def save(obj,path_name):
    with open(path_name,'wb') as file:
        pickle.dump(obj,file)

def load(path_name: object) :
    with open(path_name,'rb') as file:
        return pickle.load(file)

#自定义的dataset
class MLMTextDataset(Dataset):
    def __init__(self, examples):
        token=[]
        label=[]
        for i in examples:
            token.append('[CLS] {} [SEP] {} is a [MASK] . [SEP]'.format(i['text'],i['subj']))
            label.append(i['type'])
            token.append('[CLS] {} [SEP] {} is a [MASK] . [SEP]'.format(i['text'],i['entitys']))
            label.append(i['entityType'])
        self.tokens=token
        self.types=label

        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        

        # print(self.tokens[idx])
        # tokens = self.tokenizer.batch_encode_plus(self.tokens[idx], add_special_tokens=True,
        #                                             max_length=self.max_length, truncation=True)["input_ids"]

        # types = self.tokenizer.batch_encode_plus(self.types[idx], add_special_tokens=True,
        #                                             max_length=self.max_length, truncation=True)["input_ids"]

        # return torch.tensor(self.tokens[idx], dtype=torch.long), torch.tensor(self.tokens[idx], dtype=torch.long)
        return self.tokens[idx],str(self.tokens[idx]).replace('[MASK]',self.types[idx]),self.types[idx]

# 综合mpdel
class BertForMLM():
    def __init__(self,args):
        self.type={'FootballLeagueSeason', 'Cleric', 'Politician', 'Scientist', 'Engine', 'EducationalInstitution', 'RacingDriver', 'AmusementParkAttraction', 'Athlete', 'PeriodicalLiterature', 'SportsManager', 'Eukaryote', 'Database', 'NaturalPlace', 'Software', 'RouteOfTransportation', 'SocietalEvent', 'FloweringPlant', 'WinterSportPlayer', 'WrittenWork', 'CelestialBody', 'Organisation', 'VolleyballPlayer', 'SportsTeam', 'OrganisationMember', 'Boxer', 'Horse', 'SportsEvent', 'Broadcaster', 'Cartoon', 'ClericalAdministrativeRegion', 'Actor', 'NaturalEvent', 'MotorcycleRider', 'Genre', 'Song', 'Company', 'FictionalCharacter', 'SportsTeamSeason', 'BritishRoyalty', 'Person', 'Wrestler', 'Tower', 'RaceTrack', 'Stream', 'Satellite', 'Plant', 'Olympics', 'Tournament', 'Artist', 'Coach', 'SportFacility', 'Race', 'BodyOfWater', 'Station', 'SportsLeague', 'Presenter', 'Writer', 'LegalCase', 'Comic', 'Group', 'MusicalWork', 'Animal', 'ComicsCharacter', 'MusicalArtist', 'Building', 'Venue', 'GridironFootballPlayer', 'Infrastructure', 'Settlement',
        'CARDINAL', 'WORK_OF_ART', 'DATE', 'ORDINAL', 'LANGUAGE', 'PERSON', 'TIME', 'MONEY', 'PERCENT', 'NORP', 'EVENT', 'FAC', 'QUANTITY', 'LOC', 'PRODUCT', 'ORG', 'GPE', 'LAW'}
        self.lr=1e-5
        self.tokenizer = BertTokenizer.from_pretrained(args.bert)
        for i in self.type:
            self.tokenizer.add_tokens(i)
        self.model = BertForMaskedLM.from_pretrained(args.bert)#'/data/transformers/bert-large-cased'
        self.model.resize_token_embeddings(len(self.tokenizer))
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

def test(model,dataloader,args):
    cnt_of_right=0
    cnt_of_all=0
    with torch.no_grad():
        for i,batch in tqdm(enumerate(dataloader), desc="Testing"):
            
            tokens = model.tokenizer.batch_encode_plus(batch[0], add_special_tokens=False,
                                                         max_length=256, truncation=True,padding=True, return_tensors="pt")["input_ids"]

            tokens=tokens.to(args.device)

            outputs = model.model(tokens)

            predictions = outputs.logits.detach().cpu()
            for index,(prediction,label) in enumerate(zip(predictions,batch[2])):
                token_list=model.tokenizer.convert_ids_to_tokens(tokens[index])
                if '[MASK]' not in token_list:
                    continue
                mask_index=model.tokenizer.convert_ids_to_tokens(tokens[index]).index('[MASK]')
                
                predicted_index_mask = np.argsort(prediction[mask_index])

                tokens1=list(model.tokenizer.convert_ids_to_tokens(predicted_index_mask[-1:]))[-1]

                cnt_of_all+=1
                if tokens1==label:
                    cnt_of_right+=1
                
    return cnt_of_right/cnt_of_all


def train(args):
    pretrain_data=load(args.train_data_path)

    bertforMLM = BertForMLM(args)

    args.device=torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    bertforMLM.model.to(args.device)

    dataset=MLMTextDataset(pretrain_data)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.bsz)

    
    pretrain_data_test=load(args.test_data_path)

    dataset_test=MLMTextDataset(pretrain_data_test)

    dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=args.testBsz)


    for j in range(args.epoch):
        for i,batch in tqdm(enumerate(dataloader), desc="Training"):
            
            # todo 最多512?
            tokens = bertforMLM.tokenizer.batch_encode_plus(batch[0], add_special_tokens=False,
                                                        max_length=256, truncation=True, padding=True, return_tensors="pt")["input_ids"]

            label = bertforMLM.tokenizer.batch_encode_plus(batch[1], add_special_tokens=False,
                                                        max_length=256, truncation=True,padding=True, return_tensors="pt")["input_ids"]
            
            tokens=tokens.to(args.device)

            label=label.to(args.device)

            outputs = bertforMLM.model(tokens, labels=label)

            loss=outputs.loss

            bertforMLM.model.zero_grad()
            bertforMLM.optimizer.zero_grad()

            loss.backward()
            bertforMLM.optimizer.step()

            if(i%5000==0 and i!=0):
                print("{}||{}::{}".format(j,i,len(dataloader)))

                acc=test(bertforMLM,dataloader_test,args)

                bertforMLM.model.save_pretrained(args.model_save_dir+'{}/'.format(acc))
                bertforMLM.tokenizer.save_pretrained(args.model_save_dir+'{}/'.format(acc))
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--bsz', type=int, default=5,
                        help='batch size')
    parser.add_argument('--gpu', type=str, default="cuda:1",
                        help='batch size')
    parser.add_argument('--testBsz', type=int, default=20,
                        help='test batch size')
    parser.add_argument('--bert', type=str, required=True,
                        help='bert model')
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='pretrain_data_path')
    parser.add_argument('--test_data_path', type=str, required=True,
                        help='pretrain_test_data_path')
    parser.add_argument('--model_save_dir', type=str, required=True,
                        help='model save')
    
    args = parser.parse_args()

    train(args)
    