"""
Predict Caterpillar Tube Pricing using xgboost
"""

from __future__ import print_function

import os

import numpy as np
import pandas as pd

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CUR_DIR, 'model')


def predict_xgb(test_set):
    """Predict based on xgboost model.

    Arguments:
        test_set: str - path to test set
    """
    import xgboost as xgb

    out_file = os.path.join(CUR_DIR, 'out_xgb.csv')
    print('predicting...')

    # get test matrix
    df_in = pd.read_csv(test_set)
    test_data = df_in
    xgtest = xgb.DMatrix(test_data.values)

    # predict
    model = xgb.Booster()  # init model
    model.load_model(os.path.join(MODEL_DIR, 'model_xgb'))  # load model
    ypred = model.predict(xgtest)
    ypred = np.expm1(ypred)

    # output
    df_out = pd.DataFrame()
    df_out['id'] = np.arange(1, len(ypred) + 1)
    df_out['cost'] = ypred
    df_out.to_csv(out_file, index=False)

    return out_file


def predict_xgb_5fold(test_set):
    """Predict based on 5-fold xgboost model."""
    import xgboost as xgb

    outs = []
    print('predicting...')

    # get test matrix
    df_in = pd.read_csv(test_set)
    test_data = df_in
    xgtest = xgb.DMatrix(test_data.values)

    for fold in range(5):
        out_file = os.path.join(CUR_DIR, 'out_xgb_{}.csv'.format(fold + 1))

        # predict
        model = xgb.Booster()  # init model
        model.load_model(os.path.join(
            MODEL_DIR, 'model_xgb_{}'.format(fold + 1)))  # load model
        ypred = model.predict(xgtest)
        ypred = np.expm1(ypred)

        # output
        df_out = pd.DataFrame()
        df_out['id'] = np.arange(1, len(ypred) + 1)
        df_out['cost'] = ypred
        df_out.to_csv(out_file, index=False)
        out += [out_file]

    # ensemble
    ensemble(outs)


def predict_rf(reg, test_set):
    """Predict based on random forest model.

    Arguments:
        test_set: str - path to test set
    """
    out_file = os.path.join(CUR_DIR, 'out_rf.csv')
    print('predicting...')

    # get test matrix
    df_in = pd.read_csv(test_set)
    # log transforms for total weight
    df_in['total_weight'] = np.log1p(df_in['total_weight'])
    test_data = df_in

    # predict
    ypred = reg.predict(test_data.fillna(0).as_matrix())
    ypred = np.expm1(ypred)

    # output
    df_out = pd.DataFrame()
    df_out['id'] = np.arange(1, len(ypred) + 1)
    df_out['cost'] = ypred
    df_out.to_csv(out_file, index=False)

    return out_file


def ensemble(files, weights=None):
    """Ensembling output files."""
    print('ensembling...')
    ensemble_out = os.path.join(CUR_DIR, 'out_ens.csv')

    # validate inputs
    tmp = files
    files = [x for x in tmp if x]
    if not files:
        raise OSError
    if weights:
        assert len(files) == len(weights)
        assert np.sum(weights) == 1
    print(files, weights)

    # if only 1 file, just return the file
    if len(files) == 1:
        import shutil
        shutil.copy2(files[0], ensemble_out)
        return

    # get all files
    costs = []
    for file_ in files:
        df_ = pd.read_csv(file_, usecols=['cost'])
        costs.append(df_['cost'])

    # get ensembled cost
    ensemble_cost = pd.DataFrame()
    ensemble_cost['id'] = np.arange(1, len(costs[0]) + 1)
    ensemble_cost['cost'] = 0
    if weights:
        for cost, weight in zip(costs, weights):
            ensemble_cost['cost'] += cost * weight
    else:
        ensemble_cost['cost'] = np.mean(costs, axis=0)
    ensemble_cost.to_csv(ensemble_out, index=False)


if __name__ == '__main__':
    predict_xgb_5fold(os.path.join(MODEL_DIR, 'merged_test.csv'))
