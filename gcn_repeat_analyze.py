import argparse
from copy import copy
import json
import numpy as np
import os


def get_best_per(log_fname):
    with open(log_fname) as fin:
        lines = fin.readlines()
        for ind, line in enumerate(lines):
            line = line.replace('\'', '"')
            if line.startswith('best test:'):
                return float(line.split(' ')[2])
    return None


def report_perf_model(prefix, repeats=20):
    perfs = []
    for rep in range(repeats):
        log_fname = prefix + '{}.log'.format(rep)
        test_acc = get_best_per(log_fname)
        perfs.append(test_acc)
    return perfs, np.mean(perfs), np.std(perfs)


if __name__ == '__main__':
    desc = 'to report performance'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='citeseer',
                        help='name of dataset')
    parser.add_argument('--exp_path', type=str, default='./log/',
                        help='path of log')
    parser.add_argument('--repeats', type=int, default=20,
                        help='number of repeat data splits')
    parser.add_argument('--model', type=str, default='gcn',
                        help='number of repeat data splits')

    args = parser.parse_args()
    print(args.model, args)

    # single model
    if not args.model == 'all':
        prefix = args.model + '_' + args.dataset + '_'
        folder = 'two_layers'
        if '1' == args.model[-1]:
            folder = 'single_layer'
        prefix = os.path.join(args.exp_path, folder, prefix)
        print(prefix)
        perfs = report_perf_model(prefix, args.repeats)
        print('all:', perfs[0])
        print('mean:', perfs[1])
        print('std:', perfs[2])
        exit(0)
