#!/usr/bin/env python
from __future__ import division
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from kaggler.data_io import load_data, save_data
from kaggler.preprocessing import LabelEncoder


def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_map_file):
    logging.info('loading raw data')
    trn = pd.read_csv(train_file, index_col='id')
    tst = pd.read_csv(test_file, index_col='id')

    trn['date'] = trn.date.apply(lambda x: pd.to_datetime(x, format='%m%d%Y'))
    tst['date'] = tst.date.apply(lambda x: pd.to_datetime(x, format='%m%d%Y'))

    trn['year_2017'] = trn.date.apply(lambda x: x.year - 2016)
    tst['year_2017'] = tst.date.apply(lambda x: x.year - 2016)

    trn['month'] = trn.date.apply(lambda x: x.month)
    tst['month'] = tst.date.apply(lambda x: x.month)

    y = trn.target.values

    n_trn = trn.shape[0]

    trn.drop(['target', 'date', 'f_19'], axis=1, inplace=True)
    tst.drop(['id', 'date', 'f_19'], axis=1, inplace=True)

    cat_cols = ['customer_id'] + [x for x in trn.columns if trn[x].dtype == np.object]
    num_cols = [x for x in trn.columns if trn[x].dtype != np.object]

    logging.info('categorical: {}, numerical: {}'.format(len(cat_cols),
                                                         len(num_cols)))

    logging.info('label encoding categorical variables')
    lbe = LabelEncoder(min_obs=10)
    trn.ix[:, cat_cols] = lbe.fit_transform(trn[cat_cols].values)
    tst.ix[:, cat_cols] = lbe.transform(tst[cat_cols].values)

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(trn.columns):
            f.write('{}\t{}\tq\n'.format(i, col))

    logging.info('saving features')
    save_data(trn.values.astype(float), y, train_feature_file)
    save_data(tst.values.astype(float), None, test_feature_file)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG)

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

