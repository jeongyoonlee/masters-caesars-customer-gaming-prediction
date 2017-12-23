from const import N_CLASS
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-models', required=True, nargs='+',
                        dest='base_models')
    parser.add_argument('--feature-map-file', required=True,
                        dest='feature_map_file')

    args = parser.parse_args()

    with open(args.feature_map_file, 'w') as f:
        i = 0
        for model in enumerate(args.base_models):
            for cls in range(N_CLASS):
                f.write('{}\t{}_{}\tq\n'.format(i, model, cls))
                i += 1

