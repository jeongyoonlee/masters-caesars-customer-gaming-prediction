#!/usr/bin/env python
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time
from itertools import combinations

from kaggler.data_io import load_data, save_data
from kaggler.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


COLS_TO_DROP = ['customer_id', 'date', 'f_10', 'f_19', 'f_39']


def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_map_file):
    logging.info('loading raw data')
    trn = pd.read_csv(train_file)
    tst = pd.read_csv(test_file)

    y = trn.target.values

    n_trn = trn.shape[0]

    logging.info('adding a flag to indicate if a customer_id exists in both training and test data')
    trn['cid_both'] = trn.customer_id.isin(tst.customer_id.tolist()).astype(np.int64)
    tst['cid_both'] = tst.customer_id.isin(trn.customer_id.tolist()).astype(np.int64)
    num_cols = ['cid_both']

    logging.info('converting the date column into datetime')
    trn['date'] = pd.to_datetime(trn.date, format='%m%d%Y')
    tst['date'] = pd.to_datetime(tst.date, format='%m%d%Y')

    logging.info('add the month feature')
    trn['month'] = trn.date.dt.month
    tst['month'] = tst.date.dt.month

    logging.info('combining cid_5, month, and market')
    trn['cid_5_month_market'] = (trn.customer_id // 1e7) * 1e4 + trn.month * 100 + trn.market.str[1:].astype(int)
    tst['cid_5_month_market'] = (tst.customer_id // 1e7) * 1e4 + tst.month * 100 + tst.market.str[1:].astype(int)

    logging.info('combining cid_3, month, and market')
    trn['cid_3_month_market'] = ((trn.customer_id // 1e4) % 1e3) * 1e4 + trn.month * 100 + trn.market.str[1:].astype(int)
    tst['cid_3_month_market'] = ((tst.customer_id // 1e4) % 1e3) * 1e4 + tst.month * 100 + tst.market.str[1:].astype(int)

    cat_cols = ['cid_5_month_market', 'cid_3_month_market']

    logging.info('label encoding categorical variables')
    lbe = LabelEncoder(min_obs=10)
    trn.ix[:, cat_cols] = lbe.fit_transform(trn[cat_cols].values)
    tst.ix[:, cat_cols] = lbe.transform(tst[cat_cols].values)

    logging.info('mean-target encoding for categorical columns')
    for col in cat_cols:
        colname = 'mt_{}'.format(col)
        mean_target = trn[[col, 'target']].groupby(col).mean()
        mapping = mean_target.to_dict()['target']
        trn[colname] = trn[col].map(mapping)
        tst[colname] = tst[col].map(mapping)
        num_cols.append(colname)

    feature_cols = num_cols + cat_cols

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(feature_cols):
            if col in num_cols:
                f.write('{}\t{}\tq\n'.format(i, col))
            else:
                f.write('{}\t{}\tint\n'.format(i, col))

    logging.info('saving features')
    save_data(trn[feature_cols].values.astype(float), y, train_feature_file)
    save_data(tst[feature_cols].values.astype(float), None, test_feature_file)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')

    args = parser.parse_args()

    start = time.time()
    generate_feature(args.train_file,
                     args.test_file,
                     args.train_feature_file,
                     args.test_feature_file,
                     args.feature_map_file)
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))

