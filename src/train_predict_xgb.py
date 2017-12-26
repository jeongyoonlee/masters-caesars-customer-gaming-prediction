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

import xgboost as xgb


def train_predict(train_file, test_file, feature_map_file, predict_valid_file,
                  predict_test_file, feature_importance_file, n_est=100,
                  depth=4, lrate=.1, subcol=.5, subrow=.5, sublev=1, weight=1,
                  n_stop=100, retrain=True):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.info(('n_est={}, depth={}, lrate={}, '
                  'subcol={}, subrow={}, sublev={},'
                  'weight={}, n_stop={}').format(n_est, depth, lrate, subcol,
                                                 subrow, sublev, weight, n_stop))

    logging.info('Loading training and test data...')
    logging.info('{}'.format(model_name))
    # set xgb parameters
    params = {'objective': "reg:linear",
              'max_depth': depth,
              'eta': lrate,
              'subsample': subrow,
              'colsample_bytree': subcol,
              'colsample_bylevel': sublev,
              'min_child_weight': weight,
              'eval_metric': 'rmse',
              'silent': 1,
              'nthread': N_JOB,
              'seed': SEED}

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)
    xgtst = xgb.DMatrix(X_tst)

    logging.info('Loading CV Ids')
    cv = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    p_val = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])
    for i, (i_trn, i_val) in enumerate(cv.split(y), 1):
        xgtrn = xgb.DMatrix(X[i_trn], label=y[i_trn])
        xgval = xgb.DMatrix(X[i_val], label=y[i_val])

        logging.info('Training model #{}'.format(i))
        watchlist = [(xgtrn, 'train'), (xgval, 'val')]

        if i == 1:
            logging.info('Training with early stopping')
            clf = xgb.train(params, xgtrn, n_est, watchlist,
                            early_stopping_rounds=n_stop)
            n_best = clf.best_iteration
            logging.info('best iteration={}'.format(n_best))

            importance = clf.get_fscore(feature_map_file)
            df = pd.DataFrame.from_dict(importance, 'index')
            df.index.name = 'name'
            df.columns = ['fscore']
            df.loc[:, 'fscore'] = df.fscore / df.fscore.sum()
            df.sort_values('fscore', ascending=False, inplace=True)
            df.to_csv(feature_importance_file, index=True)
            logging.info('feature importance is saved in {}'.format(feature_importance_file))
        else:
            clf = xgb.train(params, xgtrn, n_best, watchlist)

        p_val[i_val] = clf.predict(xgval, ntree_limit=n_best)
        logging.info('CV #{}: {:.6f}'.format(i, kappa(y[i_val], p_val[i_val])))

        if not retrain:
            p_tst += clf.predict(xgtst, ntree_limit=n_best) / N_FOLD

    logging.info('CV: {:.6f}'.format(kappa(y, p_val)))
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, p_val, fmt='%.6f', delimiter=',')

    if retrain:
        logging.info('Retraining with 100% training data')
        xgtrn = xgb.DMatrix(X, label=y)
        watchlist = [(xgtrn, 'train')]
        clf = xgb.train(params, xgtrn, n_best, watchlist)
        p_tst = clf.predict(xgtst, ntree_limit=n_best)

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
    parser.add_argument('--feature-importance-file', required=True,
                        dest='feature_importance_file')
    parser.add_argument('--n-est', type=int, dest='n_est')
    parser.add_argument('--depth', type=int)
    parser.add_argument('--lrate', type=float)
    parser.add_argument('--subcol', type=float, default=1)
    parser.add_argument('--subrow', type=float, default=.5)
    parser.add_argument('--sublev', type=float, default=1.)
    parser.add_argument('--weight', type=int, default=1)
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
                  feature_map_file=args.feature_map_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  feature_importance_file=args.feature_importance_file,
                  n_est=args.n_est,
                  depth=args.depth,
                  lrate=args.lrate,
                  subcol=args.subcol,
                  subrow=args.subrow,
                  sublev=args.sublev,
                  weight=args.weight,
                  n_stop=args.n_stop,
                  retrain=args.retrain)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
