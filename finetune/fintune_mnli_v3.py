import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())

sys.path.append(CURR_DIR)
P = PATH.parent
print("current dir: ",CURR_DIR)
for i in range(1):  # add parent path, height = 3
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
import time
import copy
import random
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
import transformers
from finetune.tacred2mnli import MNLIInputFeatures
from torch.utils.data import Dataset

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
    set_global_random_seed(args.seed)
    device = torch.device("cuda:{}".format(args.cuda_index))
    args.device = device
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=3)
    if args.load_weight :
        print("load weight ",args.model_weight_path)
        model.load_state_dict(torch.load(args.model_weight_path))
    model = model.to(device)
    # optimizer = transformers.AdamW(model.parameters(),lr=4e-6)
    # schedules = transformers.get_constant_schedule_with_warmup(optimizer, )
    

    # 加载数据
    data = load(args.train_path)
    args.data_num = len(data)//3
    random.shuffle(data) # 打乱
    texts = [f"{item.premise} {tokenizer.sep_token} {item.hypothesis}."  for item in data]
    labels = [item.label for item in data]
    n_dev = int(len(data)*0.2) # dev数量
    # 划分dev, train, 创建data_loader
    dev_dataset = mnli_data(texts[:n_dev],labels[:n_dev])
    train_dataset = mnli_data(texts[n_dev:],labels[n_dev:])
    dev_loader = util_data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # create optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False) #  一般设定 4e-7 (succeeded experiment)   /2e-6 gt
    # scheduler = transformers.get_linear_schedule_with_warmup(                                    
    #     optimizer,
    #     num_warmup_steps=40,        # 40                                                  
    #     num_training_steps=args.epoch*len(train_loader)
    #     )
    # scheduler = transformers.get_constant_schedule_with_warmup( # 作者的方法                                    
    #     optimizer,
    #     num_warmup_steps=40*5,
    #     )
    print("warm up steps:",int(args.ratio*100*40))
    scheduler = transformers.get_cosine_schedule_with_warmup(                                    
        optimizer,
        num_warmup_steps=int(args.ratio*100*40),
        num_training_steps = args.epoch*(len(train_loader))
        )
    
    # train
    model.train()
    saved_paths = []
    base_acc = -1
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


            # 记录
            acc = multi_acc(prediction.detach().cpu(), label)
            accs.append(acc.item())
            losses.append(loss.detach().cpu().item())
            print('\r', " step {}/{} ,  loss_{:.4f} acc_{:.4f}  lr:{}".format(batch_idx+1,total_step,np.mean(losses),np.mean(accs),lr), end='', flush=True)
        
        train_acc  = np.mean(accs)
        train_loss = np.mean(losses)

        # evaluation
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

        # 保存模型
        args.check_point_path = os.path.join(PROJECT_PATH,"data/save_model/{}_n{}train".format(args.dataset,args.ratio)+".pt")
        if dev_acc>base_acc:
            base_acc = dev_acc
            if base_acc>0.93:
                print("save model to ",args.check_point_path)
                print("dev acc is:{}".format(dev_acc))
                torch.save(model.state_dict(),args.check_point_path)
                # 覆盖保存
                saved_paths.append(args.check_point_path)
        print()
        print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f}  dev_acc: {dev_acc:.4f}')
    # 删除不厉害的模型
    # if len(saved_paths)>=2:
    #     for paths in saved_paths[:-1]:
    #         os.remove(paths)







    


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
    # few shot models /data/tywang/O2U_model/fine_mnli_wiki_random_fewshot_0.01_epo4-0.9512.pt 


    # data1 = load("/home/tywang/myURE/URE/O2U_bert/tac_data/whole/train_top12.pkl")
    # random.seed(13)
    # index = [i for i in range(len(data1['text']))]
    # random.shuffle(index)
    # index = index[:681]
    # data1 = dict_index(data1,index)
    # save(data1,"/home/tywang/myURE/URE/O2U_bert/tac_data/random_train/0.01train_seed13.pkl")

    parser.add_argument('--batch_size', type=int, default=6, help='as named')
    parser.add_argument('--cuda_index', type=int, default=3, help='as named')
    parser.add_argument('--epoch', type=int, default=10, help='as named')
    parser.add_argument('--seed', type=int, default=16, help='as named')
    parser.add_argument('--ratio', type=float, help='as named')
    parser.add_argument('--dataset', type=str, help='as named')
    parser.add_argument('--lr', type=float, default=4e-7,
                        help='learning rate')
    # parser.add_argument("--o2u_model_path",type=str,default="/data/tywang/O2U_model",help="as named")
    # parser.add_argument('--train_path',type=str,default="/home/tywang/myURE/URE/fine_tune/out/tac_num681_only_pos_acc0.82_for_finetune_mnli.pkl")
    # parser.add_argument('--train_path',type=str,default="/home/tywang/myURE/URE/fine_tune/data/tac_num681_only_neg_acc0.98_for_finetune_mnli.pkl")
    parser.add_argument('--train_path',type=str,default="/home/tywang/myURE/URE/fine_tune/can_be_used_to_finetune/wiki_num4032_wiki_acc0.8227_ratio111_for_finetune_mnli_fewShot.pkl")
    # parser.add_argument('--dev_path',type=str,default="/home/tywang/myURE/URE/fine_tune/out/mnli_test_data.pkl")
    parser.add_argument('--model_path',type=str,default="/data/transformers/microsoft_deberta-v2-xlarge-mnli")
    parser.add_argument('--load_weight',type=str2bool)  # few-shot 模式需要加载之前finetune后的模型
    parser.add_argument('--model_weight_path',type=str,default="/data/tywang/O2U_model/fine_mnli_wiki_random_fewshot_0.05_num50_epo3-0.9333.pt",
                        help = "this is for few-shot")
    """
cd /home/tywang/myURE/URE/fine_tune
nohup python -u fintune_mnli_v3.py >/home/tywang/myURE/URE/fine_tune/log/wiki_NLNL_zeroshot_0.1.log 2>&1 &
    """
    args = parser.parse_args()
    #fine_tune_main(args)
    fine_tune_v3(args)
    """
    NLNL
    /home/tywang/myURE/URE/fine_tune/can_be_used_to_finetune/wiki_num4032_wiki_acc0.8227_ratio111_for_finetune_mnli_fewShot.pkl
    /home/tywang/myURE/URE/fine_tune/can_be_used_to_finetune/wiki_num2016_wiki_acc0.8636_ratio111_for_finetune_mnli_fewShot.pkl
    /home/tywang/myURE/URE/fine_tune/can_be_used_to_finetune/wiki_num403_wiki_acc0.9801_ratio111_for_finetune_mnli_fewShot.pkl
    """
    

    debug_stop = 1  


