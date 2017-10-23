#!/usr/bin/env python

from sklearn.model_selection import KFold
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from const import N_FOLD, SEED, N_JOB

from kaggler.data_io import load_data
from kaggler.metrics import gini

import catboost as cbt


def train_predict(train_file, test_file, feature_map_file, predict_valid_file,
                  predict_test_file, n_est=100,
                  depth=4, lrate=.1, l2_leaf_reg=1, log_file=None):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    if log_file is None:
        log_file = '{}.log'.format(model_name)

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename=log_file,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info(('n_est={}, depth={}, lrate={}, '
                  'l2_leaf_reg={}').format(n_est, depth, lrate, l2_leaf_reg))

    logging.info('Loading training and test data...')
    logging.info('{}'.format(model_name))

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    features = pd.read_csv(feature_map_file, sep='\t', header=None,
                           names=['idx', 'name', 'type'])
    cat_cols = features.idx[features.type != 'q'].tolist()

    logging.info('Loading CV Ids')
    cv = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    p_val = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])
    for i, (i_trn, i_val) in enumerate(cv.split(y), 1):

        logging.info('Training model #{}'.format(i))

        if i == 1:
            logging.info('Training with early stopping')
            clf = cbt.CatBoostRegressor(learning_rate=lrate,
                                         depth=depth,
                                         l2_leaf_reg=l2_leaf_reg,
                                         iterations=n_est,
                                         loss_function='RMSE',
                                         random_seed=SEED,
                                         thread_count=N_JOB)

            clf = clf.fit(X[i_trn], y[i_trn],
                          eval_set=[X[i_val], y[i_val]],
                          use_best_model=True,
                          cat_features=cat_cols)
            n_best = clf.tree_count_
            logging.info('best iteration={}'.format(n_best))
        else:
            clf = cbt.CatBoostRegressor(learning_rate=lrate,
                                         depth=depth,
                                         l2_leaf_reg=l2_leaf_reg,
                                         iterations=n_best,
                                         loss_function='RMSE',
                                         random_seed=SEED,
                                         thread_count=N_JOB)

            clf = clf.fit(X[i_trn], y[i_trn],
                          eval_set=(X[i_val], y[i_val]),
                          use_best_model=False,
                          cat_features=cat_cols)

        p_val[i_val] = clf.predict(X[i_val])
        logging.info('CV #{}: {:.6f}'.format(i, gini(y[i_val], p_val[i_val])))

        p_tst += clf.predict(X_tst) / N_FOLD

    logging.info('CV: {:.6f}'.format(gini(y, p_val)))
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, p_val, fmt='%.6f', delimiter=',')

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--feature-map-file', required=True,
                        dest='feature_map_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-est', type=int, dest='n_est')
    parser.add_argument('--depth', type=int)
    parser.add_argument('--lrate', type=float)
    parser.add_argument('--l2-leaf-reg', type=int, default=1, dest='l2_leaf_reg')
    parser.add_argument('--log-file', required=True, dest='log_file')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  feature_map_file=args.feature_map_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_est=args.n_est,
                  depth=args.depth,
                  lrate=args.lrate,
                  l2_leaf_reg=args.l2_leaf_reg,
                  log_file=args.log_file)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
