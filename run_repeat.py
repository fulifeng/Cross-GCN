import argparse
from copy import copy
import json
import subprocess


def get_best_para(log_fname):
    best_paras = []
    with open(log_fname) as fin:
        lines = fin.readlines()
        for ind, line in enumerate(lines):
            line = line.replace('\'', '"')
            if line.startswith('Best para:'):
                para = json.loads(line.replace('Best para: ', ''))
                best_paras.append(copy(para))
    assert len(best_paras) == 3, '%d best paras found!' % len(best_paras)
    return best_paras


def run_repeats(arguments):
    para = {}
    # load training parameters from a log file
    if arguments.source == 'load':
        best_paras = get_best_para(arguments.log)
        para = best_paras[1]
    else:
        para['dataset'] = arguments.dataset
        para['weight_decay'] = arguments.weight_decay
        para['hidden1'] = arguments.hidden1
        para['dropout'] = arguments.dropout
        para['model'] = arguments.model
        para['train_ratio'] = arguments.train_ratio
        para['epochs'] = arguments.epochs
    print(para)

    # training model along all data splits
    for rep in range(arguments.repeats):
        if arguments.source == 'load':
            ofname = arguments.log.replace('ana_tune_', '').replace('.log', '_{}.log'.format(rep))
        else:
            ofname = arguments.log.replace('.log', '_{}.log'.format(rep))
        print(ofname)
        ofile = open(ofname, 'wb')
        ofile.write(('\n\t\t' + json.dumps(para) + '\n').encode())
        if 'train_ratio' in para.keys():
            tr = str(para['train_ratio'])
        else:
            tr = '0'
        output = subprocess.check_output(
            ['python', arguments.script,
             '--dataset', para['dataset'],
             '--weight_decay', str(para['weight_decay']),
             '--hidden1', str(para['hidden1']),
             '--dropout', str(para['dropout']),
             '--model', str(para['model']),
             '--train_ratio', str(para['train_ratio']),
             '--epochs', str(para['epochs']),
             '--repeat', str(rep)
             ]
        )
        # print(output)
        ofile.write(output)
        ofile.close()


if __name__ == '__main__':
    desc = 'Run a model across different repeats'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--repeats', type=int, default=20,
                        help='number of repeats to run')
    parser.add_argument('--source', type=str, default='args',
                        choices=['args', 'load'],
                        help='source of hyper-parameter')
    parser.add_argument('--dataset', type=str, default='cora',
                        help='name of dataset')
    parser.add_argument('--model', type=str, default='gcn',
                        help='name of model')
    parser.add_argument('--log', type=str, default='./tune.log',
                        help='path and name of log file')
    parser.add_argument('--script', type=str, default='train.py',
                        help='name of the script to train model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout ratio')
    parser.add_argument('--hidden1', type=int, default=16,
                        help='number of units in hidden layer')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight for L2 loss on weights')
    parser.add_argument('--train_ratio', type=int, default=20,
                        help='number of labeled nodes per class')
    parser.add_argument('--epochs', type=int, default=200,
                        help='training epochs')

    args = parser.parse_args()
    print(args.model, args)
    run_repeats(args)
