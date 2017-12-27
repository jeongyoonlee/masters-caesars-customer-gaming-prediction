from scipy import sparse
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from kaggler.data_io import load_data, save_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-train-features', required=True, nargs='+', dest='base_train_features')
    parser.add_argument('--base-test-features', required=True, nargs='+', dest='base_test_features')
    parser.add_argument('--base-feature-maps', required=True, nargs='+', dest='base_feature_maps')
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    start = time.time()
    logging.info('combining base features for training data')
    is_sparse = False
    Xs = []
    for base_feature in args.base_train_features:
        X, y = load_data(base_feature)
        is_sparse = sparse.issparse(X) or is_sparse
        Xs.append(X)

    if is_sparse:
        X = sparse.hstack(Xs).todense()
    else:
        X = np.hstack(Xs)

    idx = np.array(X.std(axis=0) != 0).reshape(-1, )
    X = X[:, idx]
    save_data(X, y, args.train_feature_file)

    logging.info('combining base features for test data')
    Xs = []
    for base_feature in args.base_test_features:
        X, y = load_data(base_feature)
        Xs.append(X)

    if is_sparse:
        X = sparse.hstack(Xs).todense()
    else:
        X = np.hstack(Xs)

    X = X[:, idx]
    save_data(X, y, args.test_feature_file)

    logging.info('combining base feature maps')
    df = []
    for base_feature_map in args.base_feature_maps:
        feature_name = os.path.splitext(base_feature_map)[0]
        df_map = pd.read_csv(base_feature_map, sep='\t', header=None, index_col=0)
        df_map.columns = ['fname', 'ftype']
        df_map.fname = df_map.fname.apply(lambda x: '{}_{}'.format(feature_name, x))
        df.append(df_map)

    df = pd.concat(df, axis=0, ignore_index=True)
    df.iloc[idx].to_csv(args.feature_map_file, sep='\t', header=False)

    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))
