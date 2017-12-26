#!/usr/bin/env python
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy import sparse
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from kaggler.data_io import load_data, save_data
from utils import limit_mem
from const import SEED
from autoencoder import get_model


np.random.seed(SEED)
set_random_seed(SEED)


def generator(X, y=None, batch_size=1024):
    n_obs = X.shape[0]
    n_batch = int(np.ceil(n_obs / batch_size))
    while 1:
        for i in range(n_batch):
            start = i * batch_size
            end = (i + 1) * batch_size if (i + 1) * batch_size < n_obs else n_obs
            if y is not None:
                yield X[start: end].todense(), y[start: end].todense()
            else:
                yield X[start: end].todense()



def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, encoder_name, encoder_dim, lrate,
                     dropout, model_file, feature_map_file, n_est, n_stop, batch_size):
    logging.info('loading base feature files')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)
    n_trn = X.shape[0]

    logging.info('combining training and test features')
    X = sparse.vstack((X, X_tst))

    autoencoder, encoder = get_model(model_name=encoder_name,
                                     input_dim=X.shape[1],
                                     encoder_dim=encoder_dim,
                                     learning_rate=lrate,
                                     dropout=dropout)
    logging.info('training an autoencoder')
    logging.info(autoencoder.summary())

    i_trn, i_val = train_test_split(np.arange(X.shape[0]), test_size=.2,
                                    random_state=SEED, shuffle=True)

    es = EarlyStopping(monitor='val_loss', patience=n_stop)
    mcp = ModelCheckpoint(model_file, monitor='val_loss',
                          save_best_only=True, save_weights_only=False)
    h = autoencoder.fit_generator(generator(X[i_trn], X[i_trn], batch_size),
                          steps_per_epoch=int(np.ceil(len(i_trn) / batch_size)),
                          epochs=n_est,
                          validation_data=generator(X[i_val], X[i_val], batch_size),
                          validation_steps=int(np.ceil(len(i_val) / batch_size)),
                          callbacks=[es, mcp])

    val_losss = h.history['val_loss']
    n_best = val_losss.index(min(val_losss)) + 1
    autoencoder.load_weights(model_file)
    logging.info('best epoch={}'.format(n_best))

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(range(encoder_dim)):
            f.write('{}\t{}\tq\n'.format(i, col))

    logging.info('saving features')
    P = encoder.predict_generator(generator(X[:n_trn], None, batch_size),
                                  steps=int(np.ceil(n_trn / batch_size)))
    save_data(sparse.csr_matrix(P), y, train_feature_file)

    P = encoder.predict_generator(generator(X[n_trn:], None, batch_size),
                                  steps=int(np.ceil((X.shape[0] - n_trn) / batch_size)))
    save_data(sparse.csr_matrix(P), None, test_feature_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-train-feature-file', required=True, dest='train_file')
    parser.add_argument('--base-test-feature-file', required=True, dest='test_file')
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')
    parser.add_argument('--encoder-name', required=True, dest='encoder_name')
    parser.add_argument('--encoder-dim', type=int, default=128, dest='encoder_dim')
    parser.add_argument('--lrate', type=float, default=0.001, dest='lrate')
    parser.add_argument('--dropout', type=float, default=0.2, dest='dropout')
    parser.add_argument('--model-file', required=True, dest='model_file')
    parser.add_argument('--log-file', required=True, dest='log_file')
    parser.add_argument('--n-est', type=int, default=100, dest='n_est')
    parser.add_argument('--n-stop', type=int, default=10, dest='n_stop')
    parser.add_argument('--batch-size', type=int, default=1024,
            dest='batch_size')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename=args.log_file,
                        datefmt='%Y-%m-%d %H:%M:%S')

    limit_mem(args.gpu)

    start = time.time()
    generate_feature(train_file=args.train_file,
                     test_file=args.test_file,
                     train_feature_file=args.train_feature_file,
                     test_feature_file=args.test_feature_file,
                     encoder_name=args.encoder_name,
                     encoder_dim=args.encoder_dim,
                     lrate=args.lrate,
                     dropout=args.dropout,
                     model_file=args.model_file,
                     feature_map_file=args.feature_map_file,
                     n_est=args.n_est,
                     n_stop=args.n_stop,
                     batch_size=args.batch_size)
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))

