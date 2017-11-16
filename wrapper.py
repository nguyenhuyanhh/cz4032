"""
Wrapper for all steps- preprocess, train & predict
"""

import argparse
import os
from time import time

from predict import ensemble, predict_rf, predict_xgb, predict_xgb_nfold
from preprocess import preprocess
from train import train_rf, train_xgb, train_xgb_nfold

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CUR_DIR, 'model')


def _train(train_func, train_set, **kwargs):
    return train_func(train_set, **kwargs)


def _predict(predict_func, test_set, **kwargs):
    return predict_func(test_set, **kwargs)


def main(args):
    """Wrapper for everything."""
    start_time = time()

    # get datasets
    if args.no_preprocess:
        train_set = os.path.join(MODEL_DIR, 'merged_train.csv')
        test_set = os.path.join(MODEL_DIR, 'merged_test.csv')
    else:
        train_set, test_set = preprocess()

    # determine method
    if args.method == 'xgb':
        train_kwargs = {'output_model': (not args.cv), 'bayes_opt': args.bo}
        predict_kwargs = {}
        train_func, predict_func = train_xgb, predict_xgb
    elif args.method == 'rf':
        train_kwargs, predict_kwargs = {}, {}
        train_func, predict_func = train_rf, predict_rf
    elif args.method == 'xgbk':
        train_kwargs = {'n_fold': 10}
        predict_kwargs = {'n_fold': 10}
        train_func, predict_func = train_xgb_nfold, predict_xgb_nfold
    else:  # ensemble directly
        try:
            ensemble([os.path.join(CUR_DIR, 'out_xgb.csv'),
                      os.path.join(CUR_DIR, 'out_rf.csv')], [0.95, 0.05])
        except BaseException:
            pass
        return

    # train and predict
    if args.train:
        _train(train_func, train_set, **train_kwargs)
    if args.predict:
        _predict(predict_func, test_set, **predict_kwargs)

    end_time = time()
    print('exec time is {} seconds'.format(end_time - start_time))


if __name__ == '__main__':
    ARG_PARSER = argparse.ArgumentParser()
    ARG_PARSER.add_argument('method', help='method use for train and predict', choices=[
        'xgb', 'xgbk', 'rf', 'xgbrf'],)
    ARG_PARSER.add_argument('-n', '--no-preprocess',
                            action='store_true', help='skip preprocessing')
    ARG_PARSER.add_argument(
        '-p', '--predict', action='store_true', help='do predictions')
    ARG_PARSER.add_argument(
        '-t', '--train', action='store_true', help='train the model')
    ARG_PARSER.add_argument(
        '-c', '--cv', action='store_true', help='cross-validation (no model output) (only for xgb)')
    ARG_PARSER.add_argument(
        '-b', '--bo', action='store_true', help='hyper-parameter tuning using BO (only for xgb)')
    ARGS = ARG_PARSER.parse_args()
    main(ARGS)
