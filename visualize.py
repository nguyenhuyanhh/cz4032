"""
Visualize training & predicted data
"""

import pandas as pd
import os
import matplotlib.pyplot as plt


# define stuff here
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train_set.csv')


def visualize_suppliers():
    # read training file
    df_in = pd.read_csv(TRAIN_FILE)

    # calculate distribution of suppliers
    supp_dist = dict()
    for index, row in df_in.iterrows():
        # encoding for supplier
        sup = row['supplier']
        if sup not in supp_dist.keys():
            supp_dist[sup] = 1
        else:
            old_val = supp_dist[sup]
            supp_dist[sup] = old_val + 1

    # filter significant suppliers
    # take arbitrary value 100 as threshold
    new_supp_dist = dict()
    new_supp_dist['Others'] = 0
    for key,value in supp_dist.items():
        if value > 100:
            new_supp_dist[key] = value
        else:
            old_val = new_supp_dist['Others']
            new_supp_dist['Others'] = old_val + value

    plt.bar(range(len(new_supp_dist)), new_supp_dist.values(), align='center')
    plt.xticks(range(len(new_supp_dist)), list(new_supp_dist.keys()))
    plt.show()
