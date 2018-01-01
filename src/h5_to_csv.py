from kaggler.data_io import load_data, save_data
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-h5-file', required=True, dest='input_h5_file')
    parser.add_argument('--output-csv-file', required=True, dest='output_csv_file')
    args = parser.parse_args()    

    X, _ = load_data(args.input_h5_file)
    np.savetxt(args.output_csv_file, X, fmt='%f', delimiter=',')



