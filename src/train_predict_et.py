#!/usr/bin/env python

from sklearn.ensemble import ExtraTreesRegressor
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


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_est, depth, retrain=True):

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    logging.info('Loading CV Ids')
    cv = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    p_val = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])

    for i, (i_trn, i_val) in enumerate(cv.split(y), 1):
        logging.info('Training model #{}'.format(i))

        clf = ExtraTreesRegressor(n_estimators=n_est, max_depth=depth, random_state=SEED)
        clf.fit(X[i_trn], y[i_trn])
        p_val[i_val] = clf.predict(X[i_val])
        logging.info('CV #{}: {:.6f}'.format(i, kappa(y[i_val], p_val[i_val])))

        if not retrain:
            p_tst += clf.predict(X_tst) / N_FOLD

    logging.info('CV: {:.6f}'.format(kappa(y, p_val)))
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, p_val, fmt='%.6f', delimiter=',')

    if retrain:
        logging.info('Retraining with 100% training data')
        clf = ExtraTreesRegressor(n_estimators=n_est, max_depth=depth, random_state=SEED)
        clf.fit(X, y)
        p_tst = clf.predict(X_tst)

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True, dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True, dest='predict_test_file')
    parser.add_argument('--n-est', default=100, type=int)
    parser.add_argument('--depth', default=10, type=int)
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
                  depth=args.depth,
                  retrain=args.retrain)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
