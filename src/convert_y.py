import argparse
import numpy as np

def labels_exact(lb, target):
    lb = [1 if x == target else 0 for x in lb]
    return lb

def labels_less(lb, threshold):
    lb = [1 if x < threshold else 0 for x in lb]    
    return lb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-target', required=True, dest='raw_target')
    parser.add_argument('--binary-target', required=True, dest='binary_target')
    parser.add_argument('--exact-max', dest='exact_max', default=14, type=int)
    parser.add_argument('--th-max', dest='th_max', default=14, type=int)
    args = parser.parse_args()    

    y = np.loadtxt(args.raw_target, delimiter=',')

    y_binary = []

    for i in xrange(0, args.exact_max + 1):
        y_binary.append(labels_exact(y, i))
    for i in xrange(2, args.th_max + 1):
        y_binary.append(labels_less(y, i))

    y_array = np.array(y_binary)
    y_array = np.transpose(y_array)

    np.savetxt(args.binary_target, y_array, fmt='%d', delimiter=',')
