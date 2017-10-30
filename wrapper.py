"""
Wrapper for all steps- preprocess, train & predict
"""

import argparse
from time import time

from predict import predict
from preprocess import preprocess
from train import train


def main(args):
    """Wrapper for everything."""
    start_time = time()
    train_set, test_set = preprocess()
    if args.train:
        train(train_set, output_model=(not args.cv), bayes_opt=args.bo)
    if args.predict:
        predict(test_set)
    end_time = time()
    print('exec time is {} seconds'.format(end_time - start_time))


if __name__ == '__main__':
    ARG_PARSER = argparse.ArgumentParser()
    ARG_PARSER.add_argument(
        '-p', '--predict', action='store_true', help='do predictions')
    ARG_PARSER.add_argument(
        '-t', '--train', action='store_true', help='train the model')
    ARG_PARSER.add_argument(
        '-c', '--cv', action='store_true', help='do cross-validation (no model output)')
    ARG_PARSER.add_argument(
        '-b', '--bo', action='store_true', help='hyper-parameter tuning using BO')
    ARGS = ARG_PARSER.parse_args()
    main(ARGS)
