#!/usr/bin/env python

from sklearn.model_selection import KFold
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from const import N_FOLD, SEED, N_JOB
from evaluate import kappa

from kaggler.data_io import load_data

import lightgbm as lgb


np.random.seed(SEED)


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_est=100, n_leaf=200, lrate=.1, n_min=8, subcol=.3, subrow=.8,
                  subrow_freq=100, n_stop=100, retrain=True):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.info('{}'.format(model_name))
    logging.info(('n_est={}, n_leaf={}, lrate={}, '
                  'n_min={}, subcol={}, subrow={},'
                  'subrow_freq={}, n_stop={}').format(n_est, n_leaf, lrate, n_min,
                                                      subcol, subrow, subrow_freq, n_stop))

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    params = {'boosting_type': 'gbdt',
              'objective': 'regression',
              'num_leaves': n_leaf,
              'learning_rate': lrate,
              'feature_fraction': subcol,
              'bagging_fraction': subrow,
              'bagging_freq': subrow_freq,
              'min_data_in_leaf': n_min,
              'feature_franction_seed': SEED,
              'bagging_seed': SEED,
              'data_random_seed': SEED,
              'metric': 'rmse',
              'verbose': 0,
              'num_threads': N_JOB}

    logging.info('Loading CV Ids')
    cv = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    p_val = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])
    for i, (i_trn, i_val) in enumerate(cv.split(y), 1):
        logging.info('Training model #{}'.format(i))
        lgb_trn = lgb.Dataset(X[i_trn], y[i_trn])
        lgb_val = lgb.Dataset(X[i_val], y[i_val])

        if i == 1:
            logging.info('Training with early stopping')
            clf = lgb.train(params,
                            lgb_trn,
                            num_boost_round=n_est,
                            early_stopping_rounds=n_stop,
                            valid_sets=lgb_val,
                            verbose_eval=100)

            n_best = clf.best_iteration
            logging.info('best iteration={}'.format(n_best))
        else:
            clf = lgb.train(params,
                            lgb_trn,
                            num_boost_round=n_best,
                            valid_sets=lgb_val,
                            verbose_eval=100)

        p_val[i_val] = clf.predict(X[i_val])
        logging.info('CV #{}: {:.6f}'.format(i, kappa(y[i_val], p_val[i_val])))

        if not retrain:
            p_tst += clf.predict(X_tst) / N_FOLD

    logging.info('CV: {:.6f}'.format(kappa(y, p_val)))
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, p_val, fmt='%.6f', delimiter=',')

    if retrain:
        logging.info('Retraining with 100% training data')
        lgb_trn = lgb.Dataset(X, y)
        clf = lgb.train(params,
                        lgb_trn,
                        num_boost_round=n_best,
                        verbose_eval=100)

        p_tst = clf.predict(X_tst)

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-est', type=int, dest='n_est')
    parser.add_argument('--n-leaf', type=int, dest='n_leaf')
    parser.add_argument('--lrate', type=float)
    parser.add_argument('--subcol', type=float, default=1)
    parser.add_argument('--subrow', type=float, default=.5)
    parser.add_argument('--subrow-freq', type=int, default=100,
                        dest='subrow_freq')
    parser.add_argument('--n-min', type=int, default=1, dest='n_min')
    parser.add_argument('--early-stop', type=int, dest='n_stop')
    parser.add_argument('--retrain', default=False, action='store_true')
    parser.add_argument('--log-file', required=True, dest='log_file')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename=args.log_file,
                        datefmt='%Y-%m-%d %H:%M:%S')

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_est=args.n_est,
                  n_leaf=args.n_leaf,
                  lrate=args.lrate,
                  n_min=args.n_min,
                  subcol=args.subcol,
                  subrow=args.subrow,
                  subrow_freq=args.subrow_freq,
                  n_stop=args.n_stop,
                  retrain=args.retrain)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
