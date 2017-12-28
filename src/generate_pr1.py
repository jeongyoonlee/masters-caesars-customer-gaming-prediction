#!/usr/bin/env python

from __future__ import division

import argparse
import ctypes
import logging
import numpy as np
import operator
import os
import pandas as pd
import time

from const import SEED, N_FOLD
from kaggler.data_io import load_data, save_data

import lightgbm as lgb

def train_predict(train_file, test_file, binary_target_file,
                  predict_valid_file, predict_test_file,
                  cv_id_file, n_est=100, n_leaf=200, lrate=.1, n_min=8, subcol=.3, subrow=.8,
                  subrow_freq=100, n_stop=100, retrain=True):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    logging.info('Loading training and test data...')
    X, y_trn = load_data(train_file)
    X_tst, _ = load_data(test_file)

    logging.info('Loading CV Ids')
    cv_id = np.loadtxt(cv_id_file)

    logging.info('Loading binary target file')
    targets = np.loadtxt(binary_target_file, delimiter=',')
    logging.info('Targets shape {}'.format(targets.shape))

    p_vals = []
    p_tsts = []

    for col in xrange(targets.shape[1]):
        logging.info('Predict {} column of targets'.format(col))
        y = targets[:,col]

        # if col >= 2:
        #     break

        p_val = np.zeros(X.shape[0])
        p_tst = np.zeros(X_tst.shape[0])
        n_bests = []
        n_fold = N_FOLD
        for i in range(1, n_fold + 1):
            i_trn = np.where(cv_id != i)[0]
            i_val = np.where(cv_id == i)[0]
            logging.info('Training model #{}'.format(i))
            logging.debug('train: {}'.format(X[i_trn].shape))
            logging.debug('valid: {}'.format(X[i_val].shape))

            watchlist = [(X[i_val], y[i_val])]

            logging.info('Training with early stopping')
            clf = lgb.LGBMClassifier(n_estimators=n_est,
                                    num_leaves=n_leaf,
                                    learning_rate=lrate,
                                    min_child_samples=n_min,
                                    subsample=subrow,
                                    subsample_freq=subrow_freq,
                                    colsample_bytree=subcol,
                                    nthread=20,
                                    seed=SEED) 
            clf = clf.fit(X[i_trn], y[i_trn], eval_set=watchlist,
                          eval_metric="logloss", early_stopping_rounds=n_stop,
                          verbose=True)
            n_best = clf.best_iteration if clf.best_iteration > 0 else n_est
            n_bests.append(n_best)
            logging.info('best iteration={}'.format(n_best))

            p_val[i_val] = clf.predict_proba(X[i_val])[:, 1]

            if not retrain:
                p_tst += clf.predict_proba(X_tst)[:, 1] / n_fold

        if retrain:
            logging.info('Retraining with 100% training data')
            n_best = sum(n_bests) // n_fold
            clf = lgb.LGBMClassifier(n_estimators=n_best,
                                num_leaves=n_leaf,
                                learning_rate=lrate,
                                min_child_samples=n_min,
                                subsample=subrow,
                                subsample_freq=subrow_freq,
                                colsample_bytree=subcol,                                
                                nthread=20,
                                seed=SEED)
            clf = clf.fit(X, y, verbose=True)
            p_tst = clf.predict_proba(X_tst)[:, 1]

        p_vals.append(p_val)
        p_tsts.append(p_tst)

    logging.info('Saving validation predictions...')
    save_data(np.transpose(np.array(p_vals)), y_trn, predict_valid_file)

    logging.info('Saving test predictions...')
    save_data(np.transpose(np.array(p_tsts)), None, predict_test_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--binary-target-file', required=True, dest='binary_target_file')
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
    parser.add_argument('--cv-id', required=True, dest='cv_id_file')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  binary_target_file=args.binary_target_file,
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
                  retrain=args.retrain,
                  cv_id_file=args.cv_id_file)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
