from __future__ import division
from itertools import combinations
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time
from cortex.feature.feature_processing import encode_categorical_feature, log_feature_cnt, detect_column_type, dump_feature_files, fillna

from kaggler.data_io import load_data, save_data

TARGET = 'target'
EXCLUDE_COLUMNS = ['f_10', 'f_19', 'f_39', 'date', TARGET]
CAT_COLS = ['market', 'month', 'year', 'f_1', 'f_7', 'f_9','f_16', 'f_20','f_23',\
'f_24','f_29','f_31','f_33']
NUM_COLS = ['customer_id', 'customer_id_2', 'customer_id_3', 'customer_id_4', 'f_0', 'f_2', 'f_3','f_4','f_5','f_6','f_8', \
'f_11','f_12','f_13','f_14','f_15','f_17','f_18','f_21','f_22','f_25','f_26','f_27', \
'f_28','f_30','f_32','f_34','f_35','f_36','f_37','f_38','f_40','f_41']

def get_features(df):
    df["month"] = (df["date"] / 1000000).astype(int)
    df["year"] = (df["date"] % 10000).astype(int)
    df['customer_id_4'] = (df['customer_id'] / 100000000).astype(int)
    df['customer_id_3'] = (df['customer_id'] / 1000000000).astype(int)
    df['customer_id_2'] = (df['customer_id'] / 10000000000).astype(int)
    return df


def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, save_tst_target):
    logging.info('loading raw data')
    trn = pd.read_csv(train_file)
    tst = pd.read_csv(test_file)

    y_train = trn['target'].values
    
    trn_df = get_features(trn.copy())
    tst_df = get_features(tst.copy())

    categorical_columns = CAT_COLS
    numerical_columns = NUM_COLS
    
    logging.info("cat col")
    logging.info(categorical_columns)

    logging.info("num col")
    logging.info(numerical_columns)

    logging.info('null columns')
    logging.info(trn_df.columns[trn_df.isnull().any()].tolist())

    fillna(trn_df, categorical_columns, numerical_columns, num_value=-1)
    fillna(tst_df, categorical_columns, numerical_columns, num_value=-1)

    logging.info('null columns after fillna')
    logging.info(trn_df.columns[trn_df.isnull().any()].tolist())

    dump_feature_files(trn_df, tst_df, 
                       train_feature_file, test_feature_file,
                       numerical_columns, categorical_columns, TARGET, save_tst_target=save_tst_target)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    parser.add_argument('--save-tst-target', default=False, action='store_true', dest='save_tst_target')

    args = parser.parse_args()

    start = time.time()
    generate_feature(args.train_file,
                     args.test_file,
                     args.train_feature_file,
                     args.test_feature_file,
                     args.save_tst_target)
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))

