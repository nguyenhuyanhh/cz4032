"""
Preprocess data for training and testing purposes
"""

import os

import pandas as pd
pd.options.mode.chained_assignment = None

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')
MODEL_DIR = os.path.join(CUR_DIR, 'model_xgboost')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def _get_supp_encode(count=15):
    from collections import Counter

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
    df_date['year'] = tmp.str.get(0)
    df_date['month'] = tmp.str.get(1)
    df_date['date'] = tmp.str.get(2)
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
    """
    Preprocess comp_*.csv into convenient lookup tables.

    Forward lookup: component type -> component_id -> weight
    Reverse lookup: component_id -> component type
    """
    # init the component lists and lookup
    files = [i for i in os.listdir(DATA_DIR) if i[:5] == 'comp_']  # comp_*
    forward_lookup = {}
    reverse_lookup = {}
    for file_ in files:
        key = file_[5:-4]  # component type: adaptor, boss, etc.
        forward_lookup[key] = {}
        data = pd.read_csv(os.path.join(DATA_DIR, file_))
        for _, row in data.iterrows():
            reverse_lookup[row['component_id']] = key
            if row['weight'] != 'NA':
                forward_lookup[key][row['component_id']] = row['weight']
            else:
                forward_lookup[key][row['component_id']] = 0
    # handle component id 9999
    forward_lookup['other']['9999'] = 0
    reverse_lookup['9999'] = 'other'
    print('finished preprocessing components')
    return forward_lookup, reverse_lookup


def preprocess_bill_of_materials():
    """
    Preprocess bill_of_materials.csv.

    CSV header:
        tube_assembly_id,adaptor,boss,elbow,float,hfl,nut,other,sleeve,
        straight,tee,threaded,total_weight
    """
    # read bill of materials
    df_in = pd.read_csv(os.path.join(DATA_DIR, 'bill_of_materials.csv'))

    # prepare df_out
    fwd, rev = preprocess_components()
    component_types = fwd.keys()
    df_out = df_in.filter(items=['tube_assembly_id']).copy()
    df_out = df_out.reindex(columns=df_out.columns.tolist(
    ) + component_types + ['total_weight'], fill_value=0.0)

    # loop through tube_assembly_ids
    for index, row in df_in.iterrows():
        i = 1
        while i <= 8 and not pd.isnull(row['component_id_{}'.format(i)]):
            comp_type = rev[row['component_id_{}'.format(i)]]
            comp_cnt = row['quantity_{}'.format(i)]
            df_out.at[index, comp_type] += comp_cnt
            df_out.at[index,
                      'total_weight'] += fwd[comp_type][row['component_id_{}'.format(i)]] * comp_cnt
            i += 1

    print('finished preprocessing bill of materials')
    return df_out


def preprocess_specs():
    """
    Preprocess specs.csv.

    CSV header:
        tube_assembly_id, with_spec, no_spec
    """
    # read specs file
    df_in = pd.read_csv(os.path.join(DATA_DIR, 'specs.csv'))

    # create output dataframe
    df_out = df_in.filter(items=['tube_assembly_id']).copy()
    col_add = ['with_spec', 'no_spec']
    df_out = df_out.reindex(
        columns=df_out.columns.tolist() + col_add, fill_value=0)

    for index, row in df_in.iterrows():
        no_spec = row[1:].count()
        df_out.at[index, 'no_spec'] = no_spec
        if no_spec:
            df_out.at[index, 'with_spec'] = 1
        else:
            df_out.at[index, 'with_spec'] = 0

    print('finished preprocessing specs')
    return df_out


def preprocess_tube(pre_bill_of_materials, pre_specs):
    """
    Preprocessing tube.csv, with input from bill_of_materials.csv and specs.csv

    Arguments:
        pre_bill_of_materials: pd.DataFrame() - preprocessed bill of materials
        pre_specs: pd.DataFrame() - preprocessed specs
    CSV header:
        tube_assembly_id,diameter,wall,length,num_bends,bend_radius,
        end_a_1x,end_a_2x,end_x_1x,end_x_2x,end_a,end_x,adaptor,boss,
        elbow,float,hfl,nut,other,sleeve,straight,tee,threaded,total_weight,
        with_spec,no_spec
    """
    # read tube_end_form file
    df_end_form = pd.read_csv(os.path.join(DATA_DIR, 'tube_end_form.csv'))

    # encoding for end_form
    enc_form = {}
    for _, row in df_end_form.iterrows():
        if row['forming'] == 'Yes':
            enc_form[row['end_form_id']] = 1
        else:
            enc_form[row['end_form_id']] = 0
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
