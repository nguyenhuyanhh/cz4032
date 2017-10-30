"""
Preprocess data for training and testing purposes
"""

import datetime
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
SUPP_ENCODE = ['S-0066', 'S-0041', 'S-0072',
               'S-0054', 'S-0026', 'S-0013', 'S-others']
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


def preprocess_test(out_file):
    """
    Preprocess test_set.csv, with one-hot encoding for supplier and bracket

    Arguments:
        out_file: str - path to output file
    In CSV header:
        id,tube_assembly_id,supplier,quote_date,annual_usage,min_order_quantity,bracket_pricing,quantity
    Out CSV header:
        tube_assembly_id,S-0066,S-0041,S-0072,S-0054,S-0026,S-0013,S-others,
        year,month,date,annual_usage,min_order_quantity,bracket_pricing,quantity
    """
    # read training file
    df_in = pd.read_csv(TRAIN_FILE)

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
        data = pd.read_csv(os.path.join(DATA_DIR, 'file_'))
        for index, row in data.iterrows():
            reverse_lookup[row['component_id']] = key
            if row['weight'] != 'NA':
                forward_lookup[key][row['component_id']] = row['weight']
            else:
                forward_lookup[key][row['component_id']] = 0
    # handle component id 9999
    forward_lookup['other']['9999'] = '0'
    reverse_lookup['9999'] = 'other'
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
    fwd, rev = preprocess_components()
    keys = fwd.keys() + ['total_weight']
    tubes = dict()
    tmp = list()
    with open(os.path.join(DATA_DIR, 'bill_of_materials.csv'), 'r') as in_:
        tmp = in_.readlines()
    for line in tmp[1:]:
        values = line.strip().split(',')
        tubes[values[0]] = dict(zip(keys, [0] * len(keys)))
        # calculate total weight and number of each type of components
        in_ = 1
        while in_ < len(values):
            if values[in_] == 'NA':
                break
            else:
                type_ = rev[values[in_]]
                weight = int(values[in_ + 1])
                tubes[values[0]][type_] += weight
                tubes[values[0]][
                    'total_weight'] += float(fwd[type_][values[in_]]) * weight
            in_ += 2
    with open(out_file, 'w') as out_:
        out_.write(','.join(tmp[0].strip().split(
            ',')[:1] + sorted(keys)) + '\n')
        for key in sorted(tubes):
            tmp_ = [str(tubes[key][i]) for i in sorted(tubes[key].keys())]
            out_.write(','.join([key] + tmp_) + '\n')
    return out_file


def preprocess_specs(out_file):
    """
    Preprocess specs.csv.

    Arguments:
        out_file: str - path to output file
    CSV header:
        tube_assembly_id, with_spec, no_spec
    """
    tmp = list()
    with open(os.path.join(DATA_DIR, 'specs.csv'), 'r') as in_:
        tmp = in_.readlines()
    with open(out_file, 'w') as out_:
        out_.write('tube_assembly_id,with_spec,no_spec\n')
        for line in tmp[1:]:
            values = line.strip().split(',')
            tmp_ = list()
            in_ = 0
            while in_ + 1 < len(values):
                if values[in_ + 1] == 'NA':
                    break
                else:
                    in_ += 1
            if in_ == 0:
                tmp_ = [values[0], '0', str(in_)]
            else:
                tmp_ = [values[0], '1', str(in_)]
            out_.write(','.join(tmp_) + '\n')
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
    # encoding for end_form
    enc_form = dict()
    with open(os.path.join(DATA_DIR, 'tube_end_form.csv')) as in_:
        tmp_tube = in_.readlines()[1:]
        for line in tmp_tube:
            values = line.strip().split(',')
            if values[1] == 'Yes':
                enc_form[values[0]] = '1'
            else:
                enc_form[values[0]] = '0'
    enc_form['NONE'] = '0'  # handle NONE

    tmp_tube = list()  # overwrite to save memory
    tmp_bill = list()
    tmp_specs = list()
    enc_end = {'Y': '1', 'N': '0'}
    with open(TUBE_FILE, 'r') as in_:
        tmp_tube = in_.readlines()
    with open(pre_bill_of_materials, 'r') as in_:
        tmp_bill = in_.readlines()
    with open(pre_specs, 'r') as in_:
        tmp_specs = in_.readlines()
    with open(out_file, 'w') as out_:
        header = tmp_tube[0].strip().split(',')[:1] + tmp_tube[0].strip().split(',')[
            2:-1] + tmp_bill[0].strip().split(',')[1:] + tmp_specs[0].strip().split(',')[1:]
        out_.write(','.join(header) + '\n')
        in_ = 1
        while in_ < len(tmp_tube):
            v_tube = tmp_tube[in_].strip().split(',')
            v_bill = tmp_bill[in_].strip().split(',')
            v_spe = tmp_specs[in_].strip().split(',')
            content = v_tube[:1] + v_tube[2:7] + [enc_end[v_tube[i]] for i in range(7, 11)] + [
                enc_form[v_tube[i]] for i in range(11, 13)] + v_tube[13:-1] + v_bill[1:] + v_spe[1:]
            out_.write(','.join(content) + '\n')
            in_ += 1


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
    tmp_train_test = list()
    tmp_tube = list()
    with open(in_train_test_file, 'r') as in_:
        tmp_train_test = in_.readlines()
    with open(in_tube_file, 'r') as in_:
        tmp_tube = in_.readlines()
    with open(out_file, 'w') as out_:
        head_tmp = tmp_tube[0].strip().split(
            ',') + tmp_train_test[0].strip().split(',')[1:]
        out_.write(','.join(head_tmp) + '\n')
        tr_ = 1
        tu_ = 1
        while tr_ < len(tmp_train_test) and tu_ < len(tmp_tube):
            train_tmp = tmp_train_test[tr_].strip().split(',')
            tube_tmp = tmp_tube[tu_].strip().split(',')
            if train_tmp[0] < tube_tmp[0]:
                tr_ += 1
                continue
            elif train_tmp[0] > tube_tmp[0]:
                tu_ += 1
                continue
            else:
                value_tmp = tube_tmp + train_tmp[1:]
                out_.write(','.join(value_tmp) + '\n')
                tr_ += 1
                continue


def preprocess():
    """
    Wrapper for preprocessing functions.
    """
    print('preprocessing...')
    pre_train_path = os.path.join(MODEL_DIR, 'pre_train.csv')
    pre_test_path = os.path.join(MODEL_DIR, 'pre_test.csv')
    # pre_bill_path = os.path.join(MODEL_DIR, 'pre_bill_of_materials.csv')
    # pre_spec_path = os.path.join(MODEL_DIR, 'pre_specs.csv')
    # pre_tube_path = os.path.join(MODEL_DIR, 'pre_tube.csv')
    # merged_train_path = os.path.join(MODEL_DIR, 'merged_train.csv')
    # merged_test_path = os.path.join(MODEL_DIR, 'merged_test.csv')

    preprocess_train(pre_train_path)
    print('finished preprocessing train')
    preprocess_test(pre_test_path)
    print('finished preprocessing test')
    # preprocess_tube(preprocess_bill_of_materials(pre_bill_path),
    #                 preprocess_specs(pre_spec_path), pre_tube_path)
    # merge_train_test_tube(pre_train_path, pre_tube_path, merged_train_path)
    # merge_train_test_tube(pre_test_path, pre_tube_path, merged_test_path)

    # return merged_train_path, merged_test_path


preprocess()
