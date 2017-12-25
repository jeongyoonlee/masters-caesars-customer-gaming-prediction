import numpy as np
import argparse
import bisect

DISTRIBUTION = {0:62275,
1:845738,
2:720948,
3:595514,
4:474290,
5:675938,
6:437149,
7:512782,
8:281732,
9:279409,
10:127267,
11:66303,
12:70271,
13:35746,
14:16683,
15:901,
16:589,
17:270,
18:122,
19:0,
20:28}

def assignment4(expected, min_val=0, max_val=20):
    sorted_expected = sorted(expected)

    total = float(sum(DISTRIBUTION.values()))
    threshold = []
    for i in xrange(min_val, max_val):
        threshold.append( sorted_expected[int(sum([ DISTRIBUTION[x] for x in range(min_val, i+1) ]) / total * len(sorted_expected))] )

    print threshold
    
    return [ bisect.bisect_right(threshold, x) for x in expected ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file')
    parser.add_argument('-o', '--output_file')

    args = parser.parse_args()

    y_raw = np.loadtxt(args.input_file)
    y_real = assignment4(y_raw)

    np.savetxt(args.output_file, y_real, fmt='%d')