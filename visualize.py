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
VIZ_DIR = os.path.join(CUR_DIR, 'viz/')
if not os.path.exists(VIZ_DIR):
    os.makedirs(VIZ_DIR)


def visualize_suppliers():
    """Visualize the distribution of suppliers."""
    # read training file
    df_in = pd.read_csv(os.path.join(DATA_DIR, 'train_set.csv'))

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
    plt.figure()
    plt.bar(range(len(values)), values, align='center')
    plt.xticks(range(len(values)), keys, rotation=45)
    plt.suptitle('Distribution of Suppliers')
    plt.savefig(os.path.join(VIZ_DIR, 'visualize_suppliers.png'))


def visualize_specs():
    """Visualize the distribution of specs."""
    # read specs file
    df_in = pd.read_csv(os.path.join(DATA_DIR, 'specs.csv'))

    # process spec counts
    df_in['count'] = df_in[['spec{}'.format(
        i + 1) for i in range(10)]].count(axis=1)
    spec_dist = Counter(df_in['count'])

    # sort and plot
    keys = sorted(spec_dist)
    values = [spec_dist[key] for key in keys]
    plt.figure()
    plt.bar(range(len(values)), values, align='center')
    plt.xticks(range(len(values)), keys)
    plt.suptitle('Distribution of Specs')
    plt.savefig(os.path.join(VIZ_DIR, 'visualize_specs.png'))


def visualize():
    """Wrapper for all visualizations."""
    visualize_suppliers()
    visualize_specs()
    plt.show()


if __name__ == '__main__':
    visualize()
