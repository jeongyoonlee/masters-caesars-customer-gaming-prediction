#!/usr/bin/env python

from scipy import sparse
from sklearn.model_selection import KFold

import argparse
import gc
import logging
import numpy as np
import os
import pandas as pd
import time

from const import N_FOLD, SEED, N_JOB
from kaggler.data_io import load_data
from utils import limit_mem
from nn import get_model
from evaluate import kappa

from tensorflow import set_random_seed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.models import Model
from keras import backend as K


np.random.seed(SEED)
set_random_seed(SEED)


def generator(X, y=None, batch_size=1024):
    n_obs = X.shape[0]
    n_batch = int(np.ceil(n_obs / batch_size))
    while 1:
        for i in range(n_batch):
            if y is not None:
                yield X[i * batch_size: (i + 1) * batch_size], y[i * batch_size: (i + 1) * batch_size]
            else:
                yield X[i * batch_size: (i + 1) * batch_size]


def train_predict(train_file, test_file, model_file, predict_valid_file, predict_test_file,
        nn='nn2', n_est=100, lrate=.001, n_stop=100, batch_size=1024):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.info('{}'.format(model_name))
    logging.info(('{}, n_est={}, lrate={}, n_stop={}, batch_size={}').format(nn, n_est, lrate, n_stop, batch_size))

    logging.info('Loading CV Ids')
    cv = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    if sparse.issparse(X):
        X = X.todense()
        X_tst = X_tst.todense()

    logging.debug('Training ({}), and test ({}) data loaded'.format(X.shape, X_tst.shape))

    n_bests = []
    p = np.zeros_like(y, dtype=float)
    p_tst = np.zeros((X_tst.shape[0],))
    input_dim = X.shape[1]
    for i, (i_trn, i_val) in enumerate(cv.split(X, y), 1):
        logging.info('Training model #{}'.format(i))
        clf = get_model(nn, input_dim, None, lrate)
        if i == 1:
            logging.info(clf.summary())
            es = EarlyStopping(monitor='val_loss', patience=n_stop)
            mcp = ModelCheckpoint(model_file, monitor='val_loss',
                                  save_best_only=True, save_weights_only=False)
            h = clf.fit_generator(generator(X[i_trn], y[i_trn], batch_size),
                                  steps_per_epoch=int(np.ceil(len(i_trn) / batch_size)),
                                  epochs=n_est,
                                  validation_data=generator(X[i_val], y[i_val], batch_size),
                                  validation_steps=int(np.ceil(len(i_val) / batch_size)),
                                  callbacks=[es, mcp])

            val_losss = h.history['val_loss']
            n_best = val_losss.index(min(val_losss)) + 1
            clf.load_weights(model_file)
            logging.info('best epoch={}'.format(n_best))
        else:
            clf.fit_generator(generator(X[i_trn], y[i_trn], batch_size),
                              steps_per_epoch=int(np.ceil(len(i_trn) / batch_size)),
                              epochs=n_best,
                              validation_data=generator(X[i_val], y[i_val], batch_size),
                              validation_steps=int(np.ceil(len(i_val) / batch_size)))

        p[i_val] = clf.predict(X[i_val]).flatten()
        logging.info('CV {} kappa={:.6f}, best iteration={}'.format(i, kappa(y[i_val], p[i_val]), n_best))
        p_tst += clf.predict(X_tst).flatten() / N_FOLD

    logging.info('CV kappa: {:.6f}'.format(kappa(y, p)))
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, p, fmt='%.6f', delimiter=',')

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--model-file', required=True, dest='model_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--nn', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n-est', type=int, default=10, dest='n_est')
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--early-stop', type=int, dest='n_stop')
    parser.add_argument('--batch-size', type=int, dest='batch_size')
    parser.add_argument('--log-file', required=True, dest='log_file')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename=args.log_file,
                        datefmt='%Y-%m-%d %H:%M:%S')

    start = time.time()
    limit_mem(args.gpu)
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  model_file=args.model_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  nn=args.nn,
                  n_est=args.n_est,
                  lrate=args.lrate,
                  n_stop=args.n_stop,
                  batch_size=args.batch_size)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
