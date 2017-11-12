"""
Predict Caterpillar Tube Pricing using xgboost
"""

from __future__ import print_function

import os

import numpy as np
import pandas as pd

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CUR_DIR, 'model_xgboost')
# outputs
OUT_FILE = os.path.join(CUR_DIR, 'out.csv')


def predict_xgb(test_set):
    """Predict based on xgboost model.

    Arguments:
        test_set: str - path to test set
    """
    import xgboost as xgb

    print('predicting...')

    # get test matrix
    df_in = pd.read_csv(test_set)
    test_data = df_in.drop(['tube_assembly_id'], axis=1)
    xgtest = xgb.DMatrix(test_data.values)

    # predict
    model = xgb.Booster()  # init model
    model.load_model(os.path.join(MODEL_DIR, '0001.model'))  # load model
    ypred = model.predict(xgtest)
    ypred = np.expm1(ypred)

    # output
    df_out = pd.DataFrame()
    df_out['id'] = np.arange(1, len(ypred) + 1)
    df_out['cost'] = ypred
    df_out.to_csv(OUT_FILE, index=False)


def predict_rf(reg, test_set):
    """Predict based on random forest model.

    Arguments:
        test_set: str - path to test set
    """
    print('predicting...')

    # get test matrix
    df_in = pd.read_csv(test_set)
    # log transforms for total weight
    df_in['total_weight'] = np.log1p(df_in['total_weight'])
    test_data = df_in.drop(['tube_assembly_id'], axis=1)

    # predict
    ypred = reg.predict(test_data.fillna(0).as_matrix())
    ypred = np.expm1(ypred)

    # output
    df_out = pd.DataFrame()
    df_out['id'] = np.arange(1, len(ypred) + 1)
    df_out['cost'] = ypred
    df_out.to_csv(OUT_FILE, index=False)
