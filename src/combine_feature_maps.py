from scipy import sparse
import argparse
import logging
import numpy as np
import pandas as pd
import time

from kaggler.data_io import load_data, save_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-feature-maps', required=True, nargs='+', dest='base_feature_maps')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    start = time.time()
    logging.info('combining base feature maps')
    df = []
    for base_feature_map in args.base_feature_maps:
        df_map = pd.read_csv(base_feature_map, sep='\t', header=None, index_col=0)
        df_map.columns = ['fname', 'ftype']
        df.append(df_map)

    df = pd.concat(df, axis=0, ignore_index=True)
    df.to_csv(args.feature_map_file, sep='\t', header=False)
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))
