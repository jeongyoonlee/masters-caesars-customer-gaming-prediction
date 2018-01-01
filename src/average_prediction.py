#!/usr/bin/env python

from __future__ import division

import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from const import SEED
from kaggler.data_io import load_data


def train_predict(train_file, test_file, predict_valid_file, predict_test_file):
    feature_name = os.path.basename(train_file)[:-4]

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='avg_{}.log'.format(feature_name))

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    P_val = X.mean(axis=1)
    P_tst = X_tst.mean(axis=1)

    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, P_val, fmt='%.6f', delimiter=',')

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, P_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
