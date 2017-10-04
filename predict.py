"""
Predict Caterpillar Tube Pricing using xgboost
"""

from __future__ import print_function

import math
import os

import numpy as np

import xgboost as xgb

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CUR_DIR, 'model_xgboost')
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
    print('predicting...')

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
                vects[i].append(math.log(float(values[i]) + 1))
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
            # transform predictions back to y with exp(pred) - 1
            cost = math.exp(pred) - 1
            out_.write('{},{}\n'.format(id_, cost))
            id_ += 1
