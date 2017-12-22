from scipy import sparse
import argparse
import logging
import numpy as np
import time

from kaggler.data_io import load_data, save_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-features', required=True, nargs='+', dest='base_features')
    parser.add_argument('--feature-file', required=True, dest='feature_file')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    start = time.time()
    logging.info('combining base features')
    is_sparse = False
    Xs = []
    for base_feature in args.base_features:
        X, y = load_data(base_feature)
        is_sparse = sparse.issparse(X) or is_sparse
        Xs.append(X)

    if is_sparse:
        X = sparse.hstack(Xs).tocsr()
    else:
        X = np.hstack(X)

    save_data(X, y, args.feature_file)
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))
