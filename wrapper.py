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
    # specify the feature order
    feats = ['tube_assembly_id', 'diameter', 'wall', 'length', 'num_bends',
             'bend_radius', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x',
             'end_a', 'end_x',
             'adaptor', 'nut', 'sleeve', 'threaded', 'boss', 'straight',
             'elbow', 'other', 'float', 'hfl', 'tee', 'total_weight',
             'with_spec', 'no_spec',
             'S-0066', 'S-0041', 'S-0072', 'S-0054', 'S-0026', 'S-0013',
             'S-others', 'year', 'month', 'date', 'annual_usage',
             'min_order_quantity', 'bracket_pricing', 'quantity']
    # main
    start_time = time()
    train_set, test_set = preprocess()
    if args.train:
        train(feats, train_set, output_model=(not args.cv), bayes_opt=args.bo)
    if args.predict:
        predict(feats, test_set)
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
