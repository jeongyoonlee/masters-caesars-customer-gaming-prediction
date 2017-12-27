#!/usr/bin/env python
from __future__ import division
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from evaluate import kappa
from kaggler.data_io import load_data
from model import Minimizer
from const import SEED


def train_predict(train_file, test_file, predict_valid_file, predict_test_file, n_est=100):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.info('Loading training and test data...')

    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    best_seed = 42
    best_loss = np.inf
    for seed in [best_seed + x for x in range(n_est)]:
        m = Minimizer(random_state=seed)
        m.fit(X, y)
        p = m.predict(X)

        loss = -kappa(y, p)
        if loss < best_loss:
            best_loss = loss
            best_seed = seed

            logging.info('kappa = {:.4f}\nSEED = {}'.format(kappa(y, p), seed))
            logging.info('coefficients: {}'.format(m.coef_))

    logging.info('Best seed = {}'.format(best_seed))

    seed = best_seed
    m = Minimizer(random_state=seed)
    m.fit(X, y)
    p = m.predict(X)
    logging.info('kappa = {:.4f}\nSEED = {}'.format(kappa(y, p), seed))
    logging.info('coefficients: {}'.format(m.coef_))

    p_tst = m.predict(X_tst)

    logging.info('Saving predictions...')
    np.savetxt(predict_valid_file, p, fmt='%.6f', delimiter=',')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-est', default=100, type=int, dest='n_est')
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
                  n_est=args.n_est)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
