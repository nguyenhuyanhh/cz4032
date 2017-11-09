"""
Preprocess data for training and testing purposes
"""

import os

import pandas as pd

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')
MODEL_DIR = os.path.join(CUR_DIR, 'model_xgboost')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
# inputs
TRAIN_FILE = os.path.join(DATA_DIR, 'train_set.csv')
TUBE_FILE = os.path.join(DATA_DIR, 'tube.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_set.csv')
# constants
#SUPP_ENCODE = ['S-0066', 'S-0041', 'S-0072',
#               'S-0054', 'S-0026', 'S-0013', 'S-others']

# added suppliers S-0058, S-0064; Score: 0.215039
SUPP_ENCODE = ['S-0013', 'S-0026', 'S-0041', 'S-0054',
               'S-0058', 'S-0064', 'S-0066',
               'S-0072', 'S-others']

DATE_ENCODE = ['year', 'month', 'date']


def preprocess_train(out_file):
    """
    Preprocess train_set.csv, with one-hot encoding for supplier and bracket

    Arguments:
        out_file: str - path to output file
    In CSV header:
        tube_assembly_id,supplier,quote_date,annual_usage,min_order_quantity,
        bracket_pricing,quantity,cost
    Out CSV header:
        tube_assembly_id,S-0066,S-0041,S-0072,S-0054,S-0026,S-0013,S-others,
        year,month,date,annual_usage,min_order_quantity,bracket_pricing,quantity,cost
    """
    # read training file
    df_in = pd.read_csv(TRAIN_FILE)

    # create output dataframe
    col_subset = ['tube_assembly_id', 'annual_usage',
                  'min_order_quantity', 'quantity', 'cost']
    df_out = df_in.filter(items=col_subset).copy()
    col_add = SUPP_ENCODE + DATE_ENCODE
    df_out = df_out.reindex(
        columns=df_out.columns.tolist() + col_add, fill_value=0)

    for index, row in df_in.iterrows():
        # encoding for supplier
        if row['supplier'] in SUPP_ENCODE:
            df_out.at[index, row['supplier']] = 1
        else:
            df_out.at[index, SUPP_ENCODE[-1]] = 1

        # encoding for bracket
        if row['bracket_pricing'] == 'Yes':
            df_out.at[index, 'bracket_pricing'] = 1
        else:
            df_out.at[index, 'bracket_pricing'] = 0

        # encoding for date format YYYY-MM-DD
        date = row['quote_date'].split('-')
        df_out.at[index, 'year'] = date[0]
        df_out.at[index, 'month'] = date[1]
        df_out.at[index, 'date'] = date[2]

    # write to output file
    df_out.to_csv(out_file, index=False)
    print('finished preprocessing train')


def preprocess_test(out_file):
    """
    Preprocess test_set.csv, with one-hot encoding for supplier and bracket

    Arguments:
        out_file: str - path to output file
    In CSV header:
        id,tube_assembly_id,supplier,quote_date,annual_usage,min_order_quantity,
        bracket_pricing,quantity
    Out CSV header:
        tube_assembly_id,S-0066,S-0041,S-0072,S-0054,S-0026,S-0013,S-others,
        year,month,date,annual_usage,min_order_quantity,bracket_pricing,quantity
    """
    # read training file
    df_in = pd.read_csv(TEST_FILE)

    # create output dataframe
    col_subset = ['tube_assembly_id', 'annual_usage',
                  'min_order_quantity', 'quantity']
    df_out = df_in.filter(items=col_subset).copy()
    col_add = SUPP_ENCODE + DATE_ENCODE
    df_out = df_out.reindex(
        columns=df_out.columns.tolist() + col_add, fill_value=0)

    for index, row in df_in.iterrows():
        # encoding for supplier
        if row['supplier'] in SUPP_ENCODE:
            df_out.at[index, row['supplier']] = 1
        else:
            df_out.at[index, SUPP_ENCODE[-1]] = 1

        # encoding for bracket
        if row['bracket_pricing'] == 'Yes':
            df_out.at[index, 'bracket_pricing'] = 1
        else:
            df_out.at[index, 'bracket_pricing'] = 0

        # encoding for date format YYYY-MM-DD
        date = row['quote_date'].split('-')
        df_out.at[index, 'year'] = date[0]
        df_out.at[index, 'month'] = date[1]
        df_out.at[index, 'date'] = date[2]

    # write to output file
    df_out.to_csv(out_file, index=False)
    print('finished preprocessing test')


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


def preprocess_bill_of_materials(out_file):
    """
    Preprocess bill_of_materials.csv.

    Arguments:
        out_file: str - path to output file
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

    # write output
    df_out.to_csv(out_file, index=False)
    print('finished preprocessing bill of materials')

    return out_file


def preprocess_specs(out_file):
    """
    Preprocess specs.csv.

    Arguments:
        out_file: str - path to output file
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

    # write to output file
    df_out.to_csv(out_file, index=False)
    print('finished preprocessing specs')

    return out_file


def preprocess_tube(pre_bill_of_materials, pre_specs, out_file):
    """
    Preprocessing tube.csv, with input from bill_of_materials.csv and specs.csv

    Arguments:
        pre_bill_of_materials: str - path to preprocessed bill of materials
        pre_specs: str - path to preprocessed specs
        out_file: str - path to output file
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
    df_tube = pd.read_csv(TUBE_FILE)
    df_tube.drop(['material_id', 'other'], axis=1, inplace=True)
    df_tube.replace(repl_, inplace=True)

    # merge with bill_of_materials and specs
    df_bill = pd.read_csv(pre_bill_of_materials)
    df_specs = pd.read_csv(pre_specs)
    df_out = pd.merge(df_tube, df_bill, on='tube_assembly_id')
    df_out = pd.merge(df_out, df_specs, on='tube_assembly_id')

    # write to output file
    df_out.to_csv(out_file, index=False)
    print('finished preprocessing tubes')


def merge_train_test_tube(in_train_test_file, in_tube_file, out_file):
    """
    Merge two data sets from preprocess_train/test and preprocess_tube together.

    Arguments:
        in_train_test_file: str - path to input train/test file
        in_tube_file: str - path to input tube file
        out_file: str - path to output file
    CSV header:
        tube_assembly_id,[tube_contents],[train/test_contents]
    """
    # load data
    df_in = pd.read_csv(in_train_test_file)
    df_tube = pd.read_csv(in_tube_file)

    # merge
    df_out = pd.merge(df_in, df_tube, on='tube_assembly_id')

    # write output
    df_out.to_csv(out_file, index=False)
    print('finished merging')


def preprocess():
    """
    Wrapper for preprocessing functions.
    """
    print('preprocessing...')
    pre_train_path = os.path.join(MODEL_DIR, 'pre_train.csv')
    pre_test_path = os.path.join(MODEL_DIR, 'pre_test.csv')
    pre_bill_path = os.path.join(MODEL_DIR, 'pre_bill_of_materials.csv')
    pre_spec_path = os.path.join(MODEL_DIR, 'pre_specs.csv')
    pre_tube_path = os.path.join(MODEL_DIR, 'pre_tube.csv')
    merged_train_path = os.path.join(MODEL_DIR, 'merged_train.csv')
    merged_test_path = os.path.join(MODEL_DIR, 'merged_test.csv')

    preprocess_train(pre_train_path)
    preprocess_test(pre_test_path)
    preprocess_tube(preprocess_bill_of_materials(pre_bill_path),
                    preprocess_specs(pre_spec_path), pre_tube_path)
    merge_train_test_tube(pre_train_path, pre_tube_path, merged_train_path)
    merge_train_test_tube(pre_test_path, pre_tube_path, merged_test_path)

    return merged_train_path, merged_test_path
