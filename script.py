"""
A testing script file to determine how many suppliers to choose.
Constant THRES is the minimum number of data entries a supplier must have to be chosen.
"""

import os
import pandas as pd
from collections import Counter

# set Threshold for suppliers entry number
THRES = 25

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')

TRAIN_FILE = os.path.join(DATA_DIR, 'train_set.csv')

df_in = pd.read_csv(TRAIN_FILE)

my_list = df_in["supplier"].tolist()

l = Counter(my_list)
supp_l = []
for key in l.keys():
    value = l.get(key)
    if value >= THRES:
        supp_l.append(key)
print(len(supp_l))
