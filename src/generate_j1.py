#!/usr/bin/env python
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from kaggler.data_io import load_data, save_data
from kaggler.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


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

    logging.info('drop unused columns')
    trn.drop(['target', 'date', 'f_19'], axis=1, inplace=True)
    tst.drop(['id', 'date', 'f_19'], axis=1, inplace=True)

    logging.info('splitting customer ids into first 8 digits and last 4 digits')
    trn['cid_8'] = trn.customer_id // 10000
    tst['cid_8'] = tst.customer_id // 10000
    trn['cid_4'] = trn.customer_id % 10000
    tst['cid_4'] = tst.customer_id % 10000

    cat_cols = ['customer_id', 'cid_8', 'cid_4'] + [x for x in trn.columns if trn[x].dtype == np.object]
    float_cols = [x for x in trn.columns if trn[x].dtype == np.float64]
    int_cols = [x for x in trn.columns if (trn[x].dtype == np.int64) & (x not in ['cid_4', 'cid_8', 'customer_id'])]

    logging.info('categorical: {}, float: {}, int: {}'.format(len(cat_cols),
                                                              len(float_cols),
                                                              len(int_cols)))

    logging.info('label encoding categorical variables')
    lbe = LabelEncoder(min_obs=10)
    trn.ix[:, cat_cols] = lbe.fit_transform(trn[cat_cols].values)
    tst.ix[:, cat_cols] = lbe.transform(tst[cat_cols].values)

    logging.info('min-max scaling float columns')
    scaler = MinMaxScaler()
    trn.ix[:, float_cols] = scaler.fit_transform(trn[float_cols].values)
    tst.ix[:, float_cols] = scaler.transform(tst[float_cols].values)

    logging.info('adding interactions')
    interaction_cols = ['f_13', 'f_21', 'f_15', 'f_26']
    for col1, col2 in combinations(interaction_cols):
        logging.info('adding interactions between {} and {}'.format(col1, col2))
        trn['{}+{}'.format(col1, col2)] = trn[col1] + trn[col2]
        tst['{}+{}'.format(col1, col2)] = tst[col1] + tst[col2]

        trn['{}-{}'.format(col1, col2)] = trn[col1] - trn[col2]
        tst['{}-{}'.format(col1, col2)] = tst[col1] - tst[col2]

        trn['{}x{}'.format(col1, col2)] = trn[col1].apply(np.log1p) + trn[col2].apply(np.log1p)
        tst['{}x{}'.format(col1, col2)] = tst[col1].apply(np.log1p) + tst[col2].apply(np.log1p)

        trn['{}/{}'.format(col1, col2)] = trn[col1].apply(np.log1p) - trn[col2].apply(np.log1p)
        tst['{}/{}'.format(col1, col2)] = tst[col1].apply(np.log1p) - tst[col2].apply(np.log1p)

        float_cols += ['{}+{}'.format(col1, col2),
                       '{}-{}'.format(col1, col2),
                       '{}x{}'.format(col1, col2),
                       '{}/{}'.format(col1, col2)]

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(trn.columns):
            if col in cat_cols + int_cols:
                f.write('{}\t{}\tint\n'.format(i, col))
            else:
                f.write('{}\t{}\tq\n'.format(i, col))

    logging.info('saving features')
    save_data(trn.values.astype(float), y, train_feature_file)
    save_data(tst.values.astype(float), None, test_feature_file)


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

