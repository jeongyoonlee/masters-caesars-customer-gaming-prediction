#!/usr/bin/env python
from sklearn.model_selection import KFold
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time
from itertools import combinations

from const import N_FOLD, SEED, N_JOB, TARGET
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

    logging.info('splitting customer_ids into first 5 and next 3 digits')
    trn['cid_5'] = trn.customer_id // 1e7
    tst['cid_5'] = tst.customer_id // 1e7
    trn['cid_3'] = (trn.customer_id // 1e4) % 1e3
    tst['cid_3'] = (tst.customer_id // 1e4) % 1e3

    logging.info('drop unused columns')
    trn.drop(COLS_TO_DROP, axis=1, inplace=True)
    tst.drop(['id'] + COLS_TO_DROP, axis=1, inplace=True)

    cat_cols = ['cid_5', 'cid_3'] + [x for x in trn.columns if trn[x].dtype == np.object]
    float_cols = [x for x in trn.columns if trn[x].dtype == np.float64]
    int_cols = [x for x in trn.columns if (trn[x].dtype == np.int64) & (x not in ['cid_5', 'cid_3'])]

    logging.info('categorical: {}, float: {}, int: {}'.format(len(cat_cols),
                                                              len(float_cols),
                                                              len(int_cols)))

    logging.info('min-max scaling float columns')
    scaler = MinMaxScaler()
    trn.ix[:, float_cols] = scaler.fit_transform(trn[float_cols].values)
    tst.ix[:, float_cols] = scaler.transform(tst[float_cols].values)

    logging.info('adding interactions with f_5')
    interaction_cols = ['f_8', 'f_12', 'f_18', 'f_11']

    feature_cols = []
    for col in interaction_cols:
        trn['f_5+{}'.format(col)] = trn.f_5 * 10 + trn[col]
        tst['f_5+{}'.format(col)] = tst.f_5 * 10 + tst[col]
        feature_cols.append('f_5+{}'.format(col))

    for col1, col2 in combinations(interaction_cols, 2):
        logging.info('adding interactions between {} and {}'.format(col1, col2))
        trn['{}+{}'.format(col1, col2)] = trn[col1] + trn[col2]
        tst['{}+{}'.format(col1, col2)] = tst[col1] + tst[col2]

        trn['{}-{}'.format(col1, col2)] = trn[col1] - trn[col2]
        tst['{}-{}'.format(col1, col2)] = tst[col1] - tst[col2]

        trn['{}x{}'.format(col1, col2)] = trn[col1].apply(np.log1p) + trn[col2].apply(np.log1p)
        tst['{}x{}'.format(col1, col2)] = tst[col1].apply(np.log1p) + tst[col2].apply(np.log1p)

        trn['{}/{}'.format(col1, col2)] = trn[col1].apply(np.log1p) - trn[col2].apply(np.log1p)
        tst['{}/{}'.format(col1, col2)] = tst[col1].apply(np.log1p) - tst[col2].apply(np.log1p)

        feature_cols += ['{}+{}'.format(col1, col2),
                         '{}-{}'.format(col1, col2),
                         '{}x{}'.format(col1, col2),
                         '{}/{}'.format(col1, col2)]

    logging.info('saving non-CV features')
    save_data(trn[feature_cols].values.astype(float), y, train_feature_file)
    save_data(tst[feature_cols].values.astype(float), None, test_feature_file)

    logging.info('generate CV features')
    feature_name, feature_ext = os.path.splitext(train_feature_file)
    feature_name = os.path.splitext(feature_name)[0]

    cv = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    for i, (i_trn, i_val) in enumerate(cv.split(y), 1):
        cv_feature_cols = []
        logging.info('mean-target encoding for categorical columns for CV #{}'.format(i))
        cv_trn = trn[cat_cols + [TARGET]].copy()
        cv_tst = tst[cat_cols].copy()
        for col in cat_cols:
            mean_target = cv_trn.iloc[i_trn][[col, 'target']].groupby(col).mean()
            mapping = mean_target.to_dict()['target']
            cv_trn[col] = cv_trn[col].map(mapping)
            cv_tst[col] = cv_tst[col].map(mapping)

        cv_feature_cols += cat_cols

        logging.info('adding min, max, mean of mean-target encodings of categorical columns')
        cv_trn['min_target_encoding'] = cv_trn[cat_cols].min(axis=1)
        cv_trn['max_target_encoding'] = cv_trn[cat_cols].max(axis=1)
        cv_trn['median_target_encoding'] = cv_trn[cat_cols].median(axis=1)
        cv_tst['min_target_encoding'] = cv_tst[cat_cols].min(axis=1)
        cv_tst['max_target_encoding'] = cv_tst[cat_cols].max(axis=1)
        cv_tst['median_target_encoding'] = cv_tst[cat_cols].median(axis=1)

        cv_feature_cols += ['min_target_encoding', 'max_target_encoding', 'median_target_encoding']

        logging.info('saving features for CV #{}'.format(i))
        save_data(cv_trn[cv_feature_cols].values.astype(float), y, '{}.trn{}{}'.format(feature_name, i, feature_ext))
        save_data(cv_tst[cv_feature_cols].values.astype(float), None, '{}.tst{}{}'.format(feature_name, i, feature_ext))

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(feature_cols + cv_feature_cols):
            f.write('{}\t{}\tq\n'.format(i, col))


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

