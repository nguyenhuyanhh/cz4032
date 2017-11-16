"""
Train model after preprocessing training data
"""

import json
import os

import numpy as np
import pandas as pd

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CUR_DIR, 'model')


def train_xgb(train_set, config_file=os.path.join(CUR_DIR, 'config.json'),
              output_model=True, bayes_opt=False):
    """Build the xgboost model for prediction.

    Arguments:
        train_set: str - path to training set
        config_file: str - path to config file (hyper-parameters)
                default is 'config.json' in current directory
        output_model: boolean - whether to output the model
                default is True. if False, output cv score only
        bayes_opt: boolean - whether to find best parameters using BO
                default is False.
    """
    import xgboost as xgb

    # get training matrix
    df_in = pd.read_csv(train_set)
    # log transforms for cost
    df_in['cost'] = np.log1p(df_in['cost'])
    target_data = df_in['cost']
    train_data = df_in.drop(['cost'], axis=1)
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
            model = xgb.train(params, xgtrain, num_round)
            model.save_model(os.path.join(MODEL_DIR, 'model_xgb'))
        else:
            print('performing cross-validation...')
            eval_ = -xgb_evaluate()
            print('final test rmse: {}'.format(eval_))
    else:
        if output_model:
            model = xgb.train(params, xgtrain, num_round)
            model.save_model(os.path.join(MODEL_DIR, 'model_xgb'))
        else:
            print('performing cross-validation...')
            eval_ = -xgb_evaluate()
            print('final test rmse: {}'.format(eval_))


def train_xgb_nfold(train_set, n_fold=10, config_file=os.path.join(CUR_DIR, 'config.json')):
    """Train the xgboost model using n-fold."""
    from sklearn.model_selection import KFold
    import xgboost as xgb

    # xgboost parameters
    with open(config_file) as cfg:
        params = json.load(cfg)
    num_round = 5000

    # get training matrix
    df_in = pd.read_csv(train_set)
    # log transforms for cost
    df_in['cost'] = np.log1p(df_in['cost'])
    target_data = df_in['cost'].as_matrix()
    train_data = df_in.drop(['cost'], axis=1).as_matrix()

    # get n-fold and train each fold
    fold = 1
    kf_ = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    for train_index, _ in kf_.split(df_in):
        train, target = train_data[train_index], target_data[train_index]
        xgtrain = xgb.DMatrix(train, target)
        print('training fold {}...'.format(fold))
        model = xgb.train(params, xgtrain, num_round)
        model.save_model(os.path.join(MODEL_DIR, 'model_xgb_{}'.format(fold)))
        fold += 1


def train_rf(train_set):
    """Training using random forest.

    Arguments:
        train_set: str - path to training set
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.externals import joblib

    # get training matrix
    df_in = pd.read_csv(train_set)
    # log transforms cost
    df_in['cost'] = np.log1p(df_in['cost'])
    target_data = df_in['cost']
    train_data = df_in.drop(['cost'], axis=1)

    # train
    reg = RandomForestRegressor(n_jobs=-1, n_estimators=5000, verbose=1)
    reg.fit(train_data.fillna(0).as_matrix(),
            target_data.fillna(0).as_matrix())

    # save model
    joblib.dump(reg, os.path.join(MODEL_DIR, 'model_rf'))

    return reg
