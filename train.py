"""
Train model after preprocessing training data
"""

import os
import math
import numpy as np
import xgboost as xgb

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CUR_DIR, 'model_xgboost')


def train(features, train_set, output_model=True):
    """
    Build the model for prediction.

    Arguments:
        features: list(str) - list of features used to build the model
                features must match a header item in csv
        train_set: str - path to training set
        output_model: boolean - whether to output the model
                default is True. if False, output cv score only
    """
    # get training matrix
    lines = list()
    with open(train_set, 'r') as merged_:
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
            elif i == 25 or i == no_vects - 1:  # weight, cost
                vects[i].append(math.log10(float(values[i]) + 1))
            else:
                vects[i].append(float(values[i]))
    a_mat = list()
    for feat in features:
        if feat in vects_lookup.keys():
            a_mat.append(vects[vects_lookup[feat]])
    a_mat_big = np.column_stack(a_mat)

    # xgboost parameters
    dtrain = xgb.DMatrix(a_mat_big, label=vects[len(vects) - 1])
    param = {}
    param["eta"] = 0.02
    param["min_child_weight"] = 6
    param["subsample"] = 0.7
    param["colsample_bytree"] = 0.6
    param["scale_pos_weight"] = 0.8
    param["max_depth"] = 8
    param["max_delta_step"] = 2
    num_round = 5000

    # output model
    if output_model:
        model = xgb.train(param, dtrain, num_round)
        model.save_model(os.path.join(MODEL_DIR, '0001.model'))
    else:
        # using the built in cv method to check errors, it uses rmse though
        xgb.cv(param, dtrain, num_round, nfold=5, metrics={'rmse'}, seed=0, callbacks=[
            xgb.callback.print_evaluation(show_stdv=True)])
