import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(CURR_DIR)
P = PATH.parent
for i in range(3): # add parent path, height = 3
    P = P.parent
    sys.path.append(str(P.absolute()))

import json
import random
random.seed(13)
with open(os.path.join(CURR_DIR,"dev.json"),'r') as file:
    data = json.load(file)
index = [i for i in range(len(data))]
random.shuffle(index)
L = int(0.01*len(data))  # randomly select 0.01 of dev
index = index[:L]
data = [data[i] for i in index]
with open(os.path.join(CURR_DIR,"0.01dev.json"), "w") as file:
    json.dump(data, file)

debug_stop = 1