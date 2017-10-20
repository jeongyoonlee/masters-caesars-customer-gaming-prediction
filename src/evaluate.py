#!/usr/bin/env python
from __future__ import division
import argparse
import numpy as np
import os

from ml_metrics import quadratic_weighted_kappa


def kappa(y, p):
    p = np.round(p)
    p[p < 0] = 0
    p[p > 20] = 20
    return quadratic_weighted_kappa(y, p, min_rating=0, max_rating=20)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target-file', '-t', required=True, dest='target_file')
    parser.add_argument('--predict-file', '-p', required=True, dest='predict_file')

    args = parser.parse_args()

    p = np.loadtxt(args.predict_file, delimiter=',')
    y = np.loadtxt(args.target_file, delimiter=',')

    model_name = os.path.splitext(os.path.splitext(os.path.basename(args.predict_file))[0])[0]
    print('{}\t{:.6f}'.format(model_name, kappa(y, p)))
