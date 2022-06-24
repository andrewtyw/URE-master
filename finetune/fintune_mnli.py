from pprint import pprint
import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())

sys.path.append(CURR_DIR)
P = PATH.parent
print("current dir: ",CURR_DIR)
for i in range(1):  
    P = P.parent
    PROJECT_PATH = str(P.absolute())
    sys.path.append(str(P.absolute()))


from torch.utils.data import Dataset
import torch.utils.data as util_data
from tqdm import tqdm
import torch
from collections import Counter
from utils.randomness import set_global_random_seed
from utils.pickle_picky import load,save
from utils.dict_relate import dict_index
import random
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
import transformers
from torch.utils.data import Dataset
from get_mnli_pairs import from_selected

def multi_acc(y_pred, y_true):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_true).sum().float() / float(y_true.size(0))
  return acc


class mnli_data(Dataset):
    def __init__(self, texts, labels) -> object:
        self.texts = texts
        self.labels = labels
        print("各类数量:",Counter(self.labels))
        assert len(self.texts)==len(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'texts': self.texts[idx],
            'labels': self.labels[idx]}

def fine_tune_v3(args):
    print(args)
    data = from_selected(args)
    set_global_random_seed(args.seed)
    device = torch.device("cuda:{}".format(args.cuda_index))
    args.device = device
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=3)
    if args.load_weight :
        print("load weight ",args.model_weight_path)
        model.load_state_dict(torch.load(args.model_weight_path))
    model = model.to(device)
    
    
    

    
    args.data_num = len(data)//3
    random.shuffle(data) 
    
    texts = [f"{item.premise} {tokenizer.sep_token} {item.hypothesis}."  for item in data]  
    print("samples:")
    pprint(texts[:5]+texts[-5:])
    labels = [item.label for item in data]
    if not args.fewshot:
        n_dev = int(len(data)*0.2) 
        
        dev_dataset = mnli_data(texts[:n_dev],labels[:n_dev])
        train_dataset = mnli_data(texts[n_dev:],labels[n_dev:])
        dev_loader = util_data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    else:
        
        train_dataset = mnli_data(texts,labels)
        train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        dev_loader = None
    
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False) 
    
    
    
    
    
    
    
    
    
    if args.fewshot:
        warm_up_steps = len(train_loader)
    else:
        warm_up_steps = int(args.ratio*100*40)
    if args.warm_up_step is not None:
        warm_up_steps = args.warm_up_step
    print("warm up steps:",warm_up_steps)
    scheduler = transformers.get_cosine_schedule_with_warmup(                                    
        optimizer,
        num_warmup_steps=warm_up_steps,
        num_training_steps = args.epoch*(len(train_loader))
        )
    
    N_print = 1
    
    model.train()
    saved_paths = []
    base_acc = -1
    base_train_loss = 1e9
    for epoch in range(args.epoch):
        total_step = len(train_loader)
        accs = []
        losses = []
        for batch_idx, batch in enumerate(train_loader):
            text = batch['texts']
            label = batch['labels']
            
            input_ids = tokenizer.batch_encode_plus(text, padding=True, truncation=True)

            input_ids = torch.tensor(input_ids["input_ids"]).to(device)
            out = model(input_ids,labels=label.to(device))
            loss = out[0]
            prediction = out[1]
            loss.backward()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()


            
            acc = multi_acc(prediction.detach().cpu(), label)
            accs.append(acc.item())
            losses.append(loss.detach().cpu().item())
            
            
            print(" step {}/{} ,  loss_{:.4f} acc_{:.4f}  lr:{}".format(batch_idx+1,total_step,np.mean(losses),np.mean(accs),lr))
        train_acc  = np.mean(accs)
        train_loss = np.mean(losses)
        if train_loss>base_train_loss:
            break
        else:
            base_train_loss = train_loss

        if dev_loader is not None:
        
            with torch.no_grad():
                dev_accs = []
                for batch_idx, batch in enumerate(dev_loader):
                    text = batch['texts']
                    label = batch['labels']
                    input_ids = tokenizer.batch_encode_plus(text, padding=True, truncation=True)  
                    input_ids = torch.tensor(input_ids["input_ids"]).to(device)
                    prediction = model(input_ids,labels=label.to(device))[-1]
                    acc = multi_acc(prediction.detach().cpu(), label)
                    dev_accs.append(acc.detach().cpu().item())
                dev_acc = np.mean(dev_accs)

            
            
            
            if dev_acc>base_acc:
                base_acc = dev_acc
                if base_acc>0.7:
                    print("save model to ",args.check_point_path)
                    print("dev acc is:{}".format(dev_acc))
                    torch.save(model.state_dict(),args.check_point_path)
                    
                    saved_paths.append(args.check_point_path)
            print()
            print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f}  dev_acc: {dev_acc:.4f}')
        else:
            print()
            if train_acc>base_acc and train_acc>0.9:
                base_acc = train_acc
                args.check_point_path = os.path.join(PROJECT_PATH,"data/save_model/fewshot_model/{}_n{}train_{}".format(args.dataset,args.ratio,args.save_info)+".pt")
                print("save model to ",args.check_point_path)
                torch.save(model.state_dict(),args.check_point_path)
                if base_acc>0.98: break # fewshot
        if args.load_weight and epoch+1==args.epoch:
            print("load weight early break")
            break
        
        
        
                
    
    
    
    







    


    debug_stop = 1


if __name__=="__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6, help='as named')
    parser.add_argument('--cuda_index', type=int, default=0, help='as named')
    parser.add_argument('--epoch', type=int, default=10, help='as named')
    parser.add_argument('--seed', type=int, required=True, help='as named')
    parser.add_argument('--lr', type=float, default=4e-7, help='learning rate')
    parser.add_argument('--model_path',type=str,required=True)
    parser.add_argument('--load_weight',type=str2bool,default=False)  
    parser.add_argument('--model_weight_path',type=str,default=None,
                        help = "this is for few-shot")
    parser.add_argument('--check_point_path',type=str,default="",
                        help = "this is for few-shot")
    parser.add_argument('--ratio', type=float,default=0.05, help='as named')
    parser.add_argument('--dataset', type=str,required=True, help='as named')
    parser.add_argument("--label2id_path", type=str,required=True, help="as named")
    parser.add_argument("--selected_data_path", type=str,required=True, help="as named")
    parser.add_argument("--config_path", type=str,required=True, help="as named")
    parser.add_argument("--template2label_path", type=str,required=True, help="as named")

    
    parser.add_argument("--random", type=bool,default=True, help="as named")
    parser.add_argument("--fewshot", type=bool,default=False, help="as named")
    parser.add_argument("--save_info", type=str,default="", help="as named")
    parser.add_argument("--warm_up_step", type=int,default=None, help="as named")


    args = parser.parse_args()
    fine_tune_v3(args)
    


