#!/usr/bin/env python
from __future__ import division
import argparse
import numpy as np
import os

from ml_metrics import quadratic_weighted_kappa


def kappa(y, p):
    p = np.round(p).astype(int)
    p[p < 0] = 0
    p[p > 20] = 20
    return quadratic_weighted_kappa(y, p, min_rating=0, max_rating=20)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--predict-file', required=True, dest='predict_file')
    parser.add_argument('--sample-file', required=True, dest='sample_file')
    parser.add_argument('--submission-file', required=True, dest='submission_file')

    args = parser.parse_args()

    p = np.loadtxt(args.predict_file, delimiter=',')
    sub = pd.read_csv(args.sample_file, index_col=0)

    p = np.round(p).astype(int)
    p[p < 0] = 0
    p[p > 20] = 20

    sub['target'] = p
    sub.to_csv(args.submission_file)
