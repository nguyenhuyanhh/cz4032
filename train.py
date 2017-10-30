"""
Train model after preprocessing training data
"""

import json
import os

import numpy as np
import pandas as pd

import xgboost as xgb

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CUR_DIR, 'model_xgboost')


def train(train_set, config_file=os.path.join(CUR_DIR, 'config.json'),
          output_model=True, bayes_opt=False):
    """
    Build the model for prediction.

    Arguments:
        train_set: str - path to training set
        config_file: str - path to config file (hyper-parameters)
                default is 'config.json' in current directory
        output_model: boolean - whether to output the model
                default is True. if False, output cv score only
        bayes_opt: boolean - whether to find best parameters using BO
                default is False.
    """
    # get training matrix
    df_in = pd.read_csv(train_set)
    # log transforms for total weight and cost
    df_in['total_weight'] = np.log1p(df_in['total_weight'])
    df_in['cost'] = np.log1p(df_in['cost'])
    target_data = df_in['cost']
    train_data = df_in.drop(['tube_assembly_id', 'cost'], axis=1)
    xgtrain = xgb.DMatrix(train_data.values, target_data.values)

    # xgboost parameters
    with open(config_file) as cfg:
        params = json.load(cfg)
    num_round = 5000

    def xgb_evaluate(min_child_weight=None, subsample=None, colsample_bytree=None,
                     scale_pos_weight=None, max_depth=None, max_delta_step=None,
                     gamma=None):
        """Target function to evaluate model, uses xgboost's cv method."""
        # target params
        if min_child_weight is not None:
            params['min_child_weight'] = int(
                min_child_weight)  # int values only
        if subsample is not None:
            params['subsample'] = max(min(subsample, 1), 0)  # between 0 and 1
        if colsample_bytree is not None:
            params['colsample_bytree'] = max(
                min(colsample_bytree, 1), 0)  # between 0 and 1
        if scale_pos_weight is not None:
            params['scale_pos_weight'] = max(
                min(scale_pos_weight, 1), 0)  # between 0 and 1
        if max_depth is not None:
            params['max_depth'] = int(max_depth)  # int values only
        if max_delta_step is not None:
            params['max_delta_step'] = int(max_delta_step)  # int values only
        if gamma is not None:
            params['gamma'] = max(gamma, 0)  # must be positive

        # other params
        params['silent'] = 1  # silent output
        params['eta'] = 0.1

        # save params to temp
        with open(os.path.join(CUR_DIR, 'config_bo.json'), 'w') as cfg:
            json.dump(params, cfg, indent=4)

        # perform cv
        res = xgb.cv(params, xgtrain, num_boost_round=1000, nfold=5, metrics={
            'rmse'}, early_stopping_rounds=25, seed=0)
        return -res['test-rmse-mean'].values[-1]

    def xgb_bayes_opt():
        """Perform Bayesian Optimization."""
        from bayes_opt import BayesianOptimization

        print('performing hyper-parameter tuning...')

        # target params
        tests = {
            'min_child_weight': (1, 20),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.5, 1),
            'max_depth': (6, 10),
            'max_delta_step': (0, 10),
            'gamma': (0, 10)
        }

        xgb_bo = BayesianOptimization(xgb_evaluate, tests)
        xgb_bo.maximize(init_points=25, n_iter=10)

        # get the optimised params
        params_opt = xgb_bo.res['max']['max_params']
        print(params_opt)

        # run one more time to lock the values to config_bo
        xgb_bo.explore({k: [v] for k, v in params_opt.items()})

    # operations
    if bayes_opt:
        if os.path.exists(os.path.join(CUR_DIR, 'config_bo.json')):
            # BO performed before
            print('performed hyper-parameter tuning previously')
        else:
            xgb_bayes_opt()

        with open(os.path.join(CUR_DIR, 'config_bo.json'), 'r') as cfg:
            params = json.load(cfg)
        del params['silent']  # show output again

        if output_model:
            print('training...')
            model = xgb.train(params, xgtrain, num_round)
            model.save_model(os.path.join(MODEL_DIR, '0001.model'))
        else:
            print('performing cross-validation...')
            eval_ = -xgb_evaluate()
            print('final test rmse: {}'.format(eval_))
    else:
        if output_model:
            print('training...')
            model = xgb.train(params, xgtrain, num_round)
            model.save_model(os.path.join(MODEL_DIR, '0001.model'))
        else:
            print('performing cross-validation...')
            eval_ = -xgb_evaluate()
            print('final test rmse: {}'.format(eval_))


train(os.path.join(MODEL_DIR, 'merged_train.csv'), config_file=os.path.join(CUR_DIR, 'config.json'),
      output_model=True, bayes_opt=False)
