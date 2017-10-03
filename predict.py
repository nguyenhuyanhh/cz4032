"""
Predict Caterpillar Tube Pricing using xgboost
"""

from __future__ import print_function

import math
import os
from time import time

import numpy as np
import xgboost as xgb

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')
MODEL_DIR = os.path.join(CUR_DIR, 'model_xgboost')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
# inputs
TRAIN_FILE = os.path.join(DATA_DIR, 'train_set.csv')
TUBE_FILE = os.path.join(DATA_DIR, 'tube.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_set.csv')
# constants
SUPP_ENCODE = ['S-0066', 'S-0041', 'S-0072',
               'S-0054', 'S-0026', 'S-0013', 'S-others']
DATE_ENCODE = ['year', 'month', 'date']
# outputs
OUT_FILE = os.path.join(CUR_DIR, 'out.csv')


def predict(features, test_set):
    """
    Predict based on the model.

    Arguments:
        features: list(str) - list of features used to build the model
                features must match a header item in csv
        test_set: str - path to test set
    """
    # get test matrix
    lines = list()
    with open(test_set, 'r') as merged_:
        lines = merged_.readlines()
    vectors = lines[0].strip().split(',')
    no_vects = len(vectors)
    vects_lookup = {vectors[i]: i for i in range(no_vects)}
    vects = {i: list() for i in range(no_vects)}
    for line in lines[1:]:
        values = line.strip().split(',')
        for i in range(no_vects):
            if i == 0:  # id
                vects[i].append(int(values[i][-5:]))
            elif i == 25:  # weight
                vects[i].append(math.log10(float(values[i]) + 1))
            else:
                vects[i].append(float(values[i]))
    a_mat = list()
    for feat in features:
        if feat in vects_lookup.keys():
            a_mat.append(vects[vects_lookup[feat]])
    a_mat_big = np.column_stack(a_mat)

    # predict
    dtest = xgb.DMatrix(a_mat_big)
    model = xgb.Booster()  # init model
    model.load_model(os.path.join(MODEL_DIR, '0001.model'))  # load model
    ypred = model.predict(dtest)

    # output
    id_ = 1
    with open(OUT_FILE, 'w') as out_:
        out_.write('id,cost\n')
        for pred in ypred:
            cost = math.pow(10, pred) - 1
            # transform predictions back to y with (10 ** pred) - 1
            out_.write('{},{}\n'.format(id_, cost))
            id_ += 1
