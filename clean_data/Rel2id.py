import os
import sys
import copy
import random
os.chdir(sys.path[0])
sys.path.append('/root/tywang/URE')
from utils.pickle_picky import *

def rel2id(rels:list,rel2id_dict:dict=load("/root/tywang/URE/data/tacred/rel2id.pkl")):
    """
    把一个rel:list str 映射成 list int
    """
    res = [rel2id_dict[rel] if isinstance(rel,str) else rel for rel in rels]
    return res