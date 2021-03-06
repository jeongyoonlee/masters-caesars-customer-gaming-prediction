#!/usr/bin/env python

from sklearn.linear_model import LinearRegression
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


def train_predict(train_file, test_file, feature_map_file, predict_valid_file, predict_test_file,
                  feature_importance_file, retrain=True):

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    logging.info('Loading CV Ids')
    cv = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    p_val = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])

    for i, (i_trn, i_val) in enumerate(cv.split(y), 1):
        logging.info('Training model #{}'.format(i))

        clf = LinearRegression()
        clf.fit(X[i_trn], y[i_trn])
        p_val[i_val] = clf.predict(X[i_val])
        logging.info('CV #{}: {:.6f}'.format(i, kappa(y[i_val], p_val[i_val])))

        if i == 1:
            df = pd.read_csv(feature_map_file, sep='\t', names=['id', 'name', 'type'])
            df['coef'] = clf.coef_
            df.sort_values('coef', ascending=False, inplace=True)
            df.to_csv(feature_importance_file, index=False)
            logging.info('feature importance is saved in {}'.format(feature_importance_file))

        if not retrain:
            p_tst += clf.predict(X_tst) / N_FOLD

    logging.info('CV: {:.6f}'.format(kappa(y, p_val)))
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, p_val, fmt='%.6f', delimiter=',')

    if retrain:
        logging.info('Retraining with 100% training data')
        clf = LinearRegression()
        clf.fit(X, y)
        p_tst = clf.predict(X_tst)

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')
    parser.add_argument('--predict-valid-file', required=True, dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True, dest='predict_test_file')
    parser.add_argument('--feature-importance-file', required=True, dest='feature_importance_file')
    parser.add_argument('--retrain', default=False, action='store_true')
    parser.add_argument('--log-file', required=True, dest='log_file')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename=args.log_file,
                        datefmt='%Y-%m-%d %H:%M:%S')

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  feature_map_file=args.feature_map_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  feature_importance_file=args.feature_importance_file,
                  retrain=args.retrain)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
