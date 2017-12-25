#!/usr/bin/env python
from __future__ import division

import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from const import SEED, TARGET, N_FOLD

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True,
                        dest='input_file')
    parser.add_argument('--cv', '-c', required=True,
                        dest='cv_file')
    parser.add_argument('--ytrn', '-y', required=True,
                        dest='ytrn_file')
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    y = df[TARGET]

    cv = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    cv_id = np.zeros((len(y), 1))
    for i, (i_trn, i_val) in enumerate(cv.split(df, y), 1):
        cv_id[i_val] = i

    np.savetxt(args.cv_file, cv_id, fmt='%d')
    np.savetxt(args.ytrn_file, y, fmt='%d')
