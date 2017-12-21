#!/usr/bin/env python
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from kaggler.data_io import load_data, save_data
from kaggler.preprocessing import OneHotEncoder


def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_map_file):
    logging.info('loading raw data')
    trn = pd.read_csv(train_file)
    tst = pd.read_csv(test_file)

    logging.info('converting the date column into datetime')
    trn['date'] = trn.date.apply(lambda x: pd.to_datetime(x, format='%m%d%Y'))
    tst['date'] = tst.date.apply(lambda x: pd.to_datetime(x, format='%m%d%Y'))

    logging.info('add year and month features')
    trn['year_2017'] = trn.date.dt.year - 2016
    tst['year_2017'] = tst.date.dt.year - 2016

    trn['month'] = trn.date.dt.month
    tst['month'] = tst.date.dt.month

    y = trn.target.values

    n_trn = trn.shape[0]

    logging.info('splitting customer ids into first 8 digits')
    trn.customer_id = trn.customer_id // 1e7
    tst.customer_id = tst.customer_id // 1e7

    logging.info('drop unused columns')
    trn.drop(['target', 'date', 'f_19'], axis=1, inplace=True)
    tst.drop(['id', 'date', 'f_19'], axis=1, inplace=True)

    cat_cols = ['customer_id'] + [x for x in trn.columns if trn[x].dtype == np.object]
    float_cols = [x for x in trn.columns if trn[x].dtype == np.float64]
    int_cols = [x for x in trn.columns if (trn[x].dtype == np.int64) & (x != 'customer_id')]

    logging.info('categorical: {}, float: {}, int: {}'.format(len(cat_cols),
                                                              len(float_cols),
                                                              len(int_cols)))

    logging.info('label encoding categorical variables')
    ohe = OneHotEncoder(min_obs=100)
    df = pd.concat([trn, tst], axis=0)
    X_cat = ohe.fit_transform(df[int_cols + cat_cols].values)

    logging.info('min-max scaling float columns')
    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(df[float_cols].values)

    X = sparse.hstack((X_num, X_cat)).tocsr()

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(range(X.shape[1])):
            if i < X_num.shape[1]:
                f.write('{}\t{}\tq\n'.format(i, col))
            else:
                f.write('{}\t{}\ti\n'.format(i, col))

    logging.info('saving features')
    save_data(X[:n_trn], y, train_feature_file)
    save_data(X[n_trn:], None, test_feature_file)


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

