import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
print(CURR_FILE_PATH)
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(str(PATH.parent.parent.parent.parent.absolute()))
sys.path.append(str(PATH.parent.parent.parent.absolute()))
sys.path.append(str(PATH.parent.parent.absolute()))
sys.path.append(CURR_DIR)
import re
import string
import numpy as np
import copy
from URE.utils.pickle_picky import load,save
from URE.clean_data.clean import replace_s_tac,assure_replace,get_format_train_text

"""
'Tom Thabane resigned in October last year to form the All Basotho Convention Basotho Convention ( ABC ) , 
crossing the floor with 17 members of parliament ,
 causing constitutional monarch King Letsie III to dissolve parliament and call the snap election .'

 'Tom Thabane resigned in October last year to form the All Basotho Convention ( ABC ) , 
 crossing the floor with 17 members of parliament , 
 causing constitutional monarch King Letsie III to dissolve parliament and call the snap election .'



"""


wiki_data = load("/home/tywang/myURE/URE/clean_data/test_data/tac_clean_testdata.pkl")
# wiki_formatted,wiki_tags = get_format_train_text(wiki_data,mode='tac',return_tag=True)

data, tags = get_format_train_text(wiki_data,mode='tac',return_tag=True)




    

# print(sum(np.array(wiki_formatted['label'])==np.array(wiki_formatted['top1']))/len(wiki_formatted['top1']))
# # save(label2id,"/home/tywang/myURE/URE/WIKI/label2id.pkl")
# # save(wiki_formatted,"/home/tywang/myURE/URE/WIKI/test_top123.pkl")
# # save(wiki_tags,"/home/tywang/myURE/URE/WIKI/etags.pkl")
# 'Spanish shipbuilder Navantia is offering the " Cantabria " design , while South Korea \'s DSME is proposing the downsized Aegir variant of the " Tide "- class tanker .'
debug_stop = 1