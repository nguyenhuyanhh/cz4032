"""
Preprocess data for training and testing purposes
"""

import os
from collections import Counter

import pandas as pd

pd.options.mode.chained_assignment = None

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')
MODEL_DIR = os.path.join(CUR_DIR, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def _get_supp_encode(count=15):
    df_ = pd.read_csv(os.path.join(
        DATA_DIR, 'train_set.csv'), usecols=['supplier'])
    counter = Counter(df_['supplier'])

    return [x[0] for x in counter.most_common(count)]


def preprocess_train_test(train_test):
    """
    Preprocess train/test_set.csv, with one-hot encoding for supplier and bracket

    Arguments:
        train_test: boolean - specify whether is train (True) or test (False)
    In CSV header:
        tube_assembly_id,supplier,quote_date,annual_usage,min_order_quantity,
        bracket_pricing,quantity,[cost]
    Out CSV header:
        tube_assembly_id,[supplier_encoding],[quote_date_encoding],
        annual_usage,min_order_quantity,bracket_pricing,quantity,[cost]
    """
    # read source file
    if train_test:
        df_in = pd.read_csv(os.path.join(DATA_DIR, 'train_set.csv'))
        col_subset = ['annual_usage', 'min_order_quantity',
                      'bracket_pricing', 'quantity', 'cost']
        message = 'finished preprocessing train'
    else:
        df_in = pd.read_csv(os.path.join(DATA_DIR, 'test_set.csv'))
        col_subset = ['annual_usage', 'min_order_quantity',
                      'bracket_pricing', 'quantity']
        message = 'finished preprocessing test'

    # encoding for supplier
    supp_encode = _get_supp_encode()
    df_supp = df_in[['tube_assembly_id', 'supplier']]
    for supp in supp_encode:
        df_supp[supp] = (df_supp['supplier'] == supp).astype(int)
    df_supp['S-others'] = 1
    df_supp['S-others'] = df_supp['S-others'] - \
        df_supp[supp_encode].sum(axis=1)
    df_supp.drop(['supplier'], axis=1, inplace=True)

    # encoding for date
    df_date = df_in[['quote_date']]
    tmp = df_date['quote_date'].str.split('-')
    df_date['year'] = tmp.str.get(0).astype(int)
    df_date['month'] = tmp.str.get(1).astype(int)
    df_date['date'] = tmp.str.get(2).astype(int)
    df_date.drop(['quote_date'], axis=1, inplace=True)

    # encoding for bracket_pricing'
    enc_ = {'bracket_pricing': {'Yes': 1, 'No': 0}}

    # create output dataframe
    df_out = pd.concat([df_supp, df_date], axis=1)
    df_out = pd.concat([df_out, df_in[col_subset]], axis=1)
    df_out.replace(enc_, inplace=True)

    print(message)
    return df_out


def preprocess_components():
    """Preprocess components.

    CSV header:
        component_id, component_type, unique_feature,
        orientation, weight
    """
    # get features
    files = [i for i in os.listdir(DATA_DIR) if i[:5] == 'comp_']  # comp_*
    dfs = []
    for file_ in files:
        df_ = pd.read_csv(os.path.join(DATA_DIR, file_))
        df_tmp = pd.DataFrame()
        df_tmp['component_id'] = df_['component_id']
        df_tmp['component_type'] = file_[5:-4]
        try:
            df_tmp['unique_feature'] = df_['unique_feature']
        except KeyError:  # no unique_feature
            df_tmp['unique_feature'] = 'No'
        try:
            df_tmp['orientation'] = df_['orientation']
        except KeyError:  # no orientation
            df_tmp['orientation'] = 'No'
        df_tmp['weight'] = df_['weight']
        dfs += [df_tmp]

    # produce output
    df_out = pd.concat(dfs, ignore_index=True)
    df_out.replace({'unique_feature': {'Yes': 1, 'No': 0},
                    'orientation': {'Yes': 1, 'No': 0}}, inplace=True)

    # handle component 9999 and 0 for later
    app = [
        {'component_id': '9999', 'weight': 0,
         'component_type': 'other', 'unique_feature': 0,
         'orientation': 0},
        {'component_id': 0, 'weight': 0,
         'component_type': 'type_0', 'unique_feature': 0,
         'orientation': 0}
    ]
    df_out = df_out.append(app, ignore_index=True)
    print('finished preprocessing components')
    return df_out


def preprocess_bill_of_materials():
    """Preprocess bill of materials.

    CSV header:
        tube_assembly_id,[component_type_encoding],
        [unique_feature_encoding],[orientation_encoding]
        total_weight
    """
    # read bill of materials, fill NaNs with 0s
    df_in = pd.read_csv(os.path.join(
        DATA_DIR, 'bill_of_materials.csv')).fillna(0)

    # load components
    df_comp = preprocess_components()
    comp_types = Counter(df_comp['component_type']).keys()

    # init encodings and weight
    for comp_type in comp_types:
        df_in[comp_type] = 0
    df_in['unique_feature'] = 0
    df_in['orientation'] = 0
    df_in['total_weight'] = 0

    # calculate
    drops = []
    for i in range(8):
        # merge with components
        cid = 'component_id_{}'.format(i + 1)
        wei = 'weight_{}'.format(i + 1)
        ctp = 'component_type_{}'.format(i + 1)
        unq = 'unique_feature_{}'.format(i + 1)
        ori = 'orientation_{}'.format(i + 1)
        qty = 'quantity_{}'.format(i + 1)
        df_comp[cid], df_comp[wei], df_comp[ctp], df_comp[unq], df_comp[ori] = \
            df_comp['component_id'], df_comp['weight'], df_comp['component_type'], \
            df_comp['unique_feature'], df_comp['orientation']
        df_in = df_in.merge(df_comp[[cid, wei, ctp, unq, ori]],
                            on=cid, sort=False)
        # weight
        df_in['total_weight'] += df_in[qty] * df_in[wei]
        # unique_feature
        df_in['unique_feature'] += df_in[qty] * df_in[unq]
        # orientation
        df_in['orientation'] += df_in[qty] * df_in[ori]
        # component_type_encoding
        for comp_type in comp_types:
            df_in[comp_type] += ((df_in[ctp] ==
                                  comp_type).astype(int)) * df_in[qty]
        # drop columns later
        drops += [cid, wei, ctp, unq, ori, qty]

    df_in.drop(drops + ['type_0'], axis=1, inplace=True)
    print('finished preprocessing bill of materials')
    return df_in


def preprocess_specs():
    """
    Preprocess specs.csv.

    CSV header:
        tube_assembly_id, with_spec, no_spec
    """
    # read specs file
    df_in = pd.read_csv(os.path.join(DATA_DIR, 'specs.csv'))
    spec_cols = ['spec{}'.format(i + 1) for i in range(10)]

    # with_spec and no_spec
    df_in['with_spec'] = 0
    df_in['no_spec'] = df_in[spec_cols].count(axis=1)
    df_in['with_spec'] = (df_in['no_spec'] > 0).astype(int)
    df_in.drop(spec_cols, axis=1, inplace=True)

    print('finished preprocessing specs')
    return df_in


def preprocess_tube(pre_bill_of_materials, pre_specs):
    """
    Preprocessing tube.csv, with input from bill_of_materials.csv and specs.csv

    Arguments:
        pre_bill_of_materials: pd.DataFrame() - preprocessed bill of materials
        pre_specs: pd.DataFrame() - preprocessed specs
    CSV header:
        tube_assembly_id,[bill_of_materials_encoding],[specs_encoding]
    """
    # read tube_end_form file
    df_end_form = pd.read_csv(os.path.join(DATA_DIR, 'tube_end_form.csv'))

    # encoding for end_form
    enc_form = {}
    for _, row in df_end_form.iterrows():
        if row['forming'] == 'Yes':
            enc_form[row['end_form_id']] = 2
        else:
            enc_form[row['end_form_id']] = 1
    enc_form['NONE'] = 0  # handle NONE

    # process tube in-memory
    repl_ = {k: {'Y': 1, 'N': 0}
             for k in ['end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x']}
    repl_['end_a'] = enc_form
    repl_['end_x'] = enc_form
    df_tube = pd.read_csv(os.path.join(DATA_DIR, 'tube.csv'))
    df_tube.drop(['material_id', 'other'], axis=1, inplace=True)
    df_tube.replace(repl_, inplace=True)

    # merge with bill_of_materials and specs
    df_out = pd.merge(df_tube, pre_bill_of_materials, on='tube_assembly_id')
    df_out = pd.merge(df_out, pre_specs, on='tube_assembly_id')

    print('finished preprocessing tubes')
    return df_out


def merge_train_test_tube(pre_train_test, pre_tube, out_file):
    """
    Merge two data sets from preprocess_train/test and preprocess_tube together.

    Arguments:
        pre_train_test: pd.DataFrame() - preprocessed train/test
        pre_tube: pd.DataFrame() - preprocessed tube
        out_file: str - path to output file
    CSV header:
        tube_assembly_id,[tube_contents],[train/test_contents]
    """
    # merge
    df_out = pd.merge(pre_train_test, pre_tube, on='tube_assembly_id')
    df_out['tube_assembly_id'] = df_out['tube_assembly_id'].str.split(
        '-').str.get(1).astype(int)

    # write output
    df_out.to_csv(out_file, index=False)
    print('finished merging')


def preprocess():
    """
    Wrapper for preprocessing functions.
    """
    print('preprocessing...')
    merged_train_path = os.path.join(MODEL_DIR, 'merged_train.csv')
    merged_test_path = os.path.join(MODEL_DIR, 'merged_test.csv')

    pre_train = preprocess_train_test(True)
    pre_test = preprocess_train_test(False)
    pre_tube = preprocess_tube(
        preprocess_bill_of_materials(), preprocess_specs())
    merge_train_test_tube(pre_train, pre_tube, merged_train_path)
    merge_train_test_tube(pre_test, pre_tube, merged_test_path)

    return merged_train_path, merged_test_path


if __name__ == '__main__':
    print(preprocess())
