"""
Visualize training & predicted data
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

# define stuff here
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train_set.csv')


def visualize_suppliers():
    """Visualize the distribution of suppliers."""
    # read training file
    df_in = pd.read_csv(TRAIN_FILE)

    # calculate distribution of suppliers
    supp_dist = Counter(df_in['supplier'])

    # filter significant suppliers
    # take arbitrary value 100 as threshold
    new_supp_dist = {}
    new_supp_dist['Others'] = 0
    for key, value in supp_dist.items():
        if value > 100:
            new_supp_dist[key] = value
        else:
            new_supp_dist['Others'] += value

    # sort and plot
    values = sorted(new_supp_dist.values(), reverse=True)
    keys = sorted(new_supp_dist, key=new_supp_dist.get, reverse=True)
    plt.bar(range(len(values)), values, align='center')
    plt.xticks(range(len(values)), keys)
    plt.show()


if __name__ == '__main__':
    visualize_suppliers()
