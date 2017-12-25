from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa as kappa
import argparse
import logging
import numpy as np
import pandas as pd
import time


def preds_to_digits(cutoffs, preds):
    res = []
    for p in list(preds):
        if p < cutoffs[0]:
            res.append(0)
        elif p < cutoffs[1]:
            res.append(1)
        elif p < cutoffs[2]:
            res.append(2)
        elif p < cutoffs[3]:
            res.append(3)
        elif p < cutoffs[4]:
            res.append(4)
        elif p < cutoffs[5]:
            res.append(5)
        elif p < cutoffs[6]:
            res.append(6)
        elif p < cutoffs[7]:
            res.append(7)
        elif p < cutoffs[8]:
            res.append(8)
        elif p < cutoffs[9]:
            res.append(9)
        elif p < cutoffs[10]:
            res.append(10)
        elif p < cutoffs[11]:
            res.append(11)
        elif p < cutoffs[12]:
            res.append(12)
        elif p < cutoffs[13]:
            res.append(13)
        elif p < cutoffs[14]:
            res.append(14)
        elif p < cutoffs[15]:
            res.append(15)
        else:
            res.append(16)

    return np.array(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict-file', required=True, dest='predict_file')
    parser.add_argument('--sample-file', required=True, dest='sample_file')
    parser.add_argument('--submission-file', required=True, dest='submission_file')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    start = time.time()
    logging.info('writing a submission file by applying the cutoffs to test predictions')
    x0 = (1.5, 2.2, 3.0, 3.8, 4.5, 5.2, 6.0, 6.8, 7.5, 8.3, 9.1, 9.9, 10.7, 11.6, 12.4, 13.2)
    p_tst = np.loadtxt(args.predict_file)
    sub = pd.read_csv(args.sample_file, index_col=0)
    sub.target = preds_to_digits(x0, p_tst)
    sub.to_csv(args.submission_file)
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))
