import argparse
import os
import numpy as np
import torch

from sklearn.metrics import precision_recall_fscore_support
from torchlight.torchlight.gpu import ngpu

from utils import loader, processor, common

base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path, '../data')
ftype = ''
coords = 3
joints = 16
cycles = 1
num_inits = 10
num_folds = 10

parser = argparse.ArgumentParser(description='Gait Gen')
parser.add_argument('--train', type=bool, default=True, metavar='T',
                    help='train the model (default: True)')
parser.add_argument('--batch-size', type=int, default=8, metavar='B',
                    help='input batch size for training (default: 8)')
parser.add_argument('--num-worker', type=int, default=4, metavar='W',
                    help='input batch size for training (default: 4)')
parser.add_argument('--start_epoch', type=int, default=0, metavar='SE',
                    help='starting epoch of training (default: 0)')
parser.add_argument('--num_epoch', type=int, default=500, metavar='NE',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                    help='optimizer (default: SGD)')
parser.add_argument('--base-lr', type=float, default=0.001, metavar='L',
                    help='base learning rate (default: 0.001)')
parser.add_argument('--step', type=list, default=[0.5, 0.75, 0.875], metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: [0.5, 0.75, 0.875])')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='D',
                    help='Weight decay (default: 1e-4)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is evaluated (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='interval after which log is printed (default: 100)')
parser.add_argument('--topk', type=list, default=[1], metavar='[K]',
                    help='top K accuracy to show (default: [1])')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pavi-log', action='store_true', default=False,
                    help='pavi log')
parser.add_argument('--print-log', action='store_true', default=True,
                    help='print log')
parser.add_argument('--save-log', action='store_true', default=True,
                    help='save log')

args = parser.parse_args()
device = 'cuda:0'

datasets = ['BML', 'CMU', 'Human3.6M', 'ICT', 'SIG', 'UNC_RGB']

data, labels, data_train_all_folds, labels_train_all_folds,\
    data_test_all_folds, labels_test_all_folds = \
        loader.load_data(data_path, ftype, joints, coords, cycles=cycles, num_folds=num_folds)

metrics_file_full_path = 'metrics.txt'
if not os.path.exists(metrics_file_full_path):
    for init_idx in range(num_inits):
        for fold_idx, (data_train, labels_train, data_test, labels_test) in enumerate(zip(data_train_all_folds, labels_train_all_folds,
                                                                                        data_test_all_folds, labels_test_all_folds)):
            print('Running init {:02d}, fold {:02d}'.format(init_idx, fold_idx))
            # saving trained models for each init and split in separate folders
            model_path = os.path.join(base_path, 'model_classifier_combined_lstm_init_{:02d}_fold_{:02d}/features'.format(init_idx, fold_idx) + ftype)
            args.work_dir = model_path
            os.makedirs(model_path, exist_ok=True)
            aff_features = len(data_train[0][0])
            num_classes = np.unique(labels_train).shape[0]
            data_loader = list()
            data_loader.append(torch.utils.data.DataLoader(
                dataset=loader.TrainTestLoader(data_train, labels_train, joints, coords),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_worker * ngpu(device),
                drop_last=True))
            data_loader.append(torch.utils.data.DataLoader(
                dataset=loader.TrainTestLoader(data_test, labels_test, joints, coords),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_worker * ngpu(device),
                drop_last=True))
            data_loader = dict(train=data_loader[0], test=data_loader[1])
            graph_dict = {'strategy': 'spatial'}
            pr = processor.Processor(args, data_loader, coords*joints, aff_features, num_classes, graph_dict, device=device)
            if args.train:
                pr.train()

            best_features, label_preds = pr.extract_best_feature(data_test, joints, coords)
            # print('{:.4f}'.format(sum(labels == label_preds)/labels.shape[0]))
            # common.plot_features(best_features, labels)

            precision, recall, fscore, _ = precision_recall_fscore_support(labels_test, label_preds, average='weighted')
            accuracy = sum(labels_test == label_preds) / np.shape(labels_test)[0]
            # accuracy = '{:.4f}'.format(sum(labels_test == label_preds)/np.shape(labels_test)[0])
            print(precision, recall, fscore, accuracy)
            metrics_file_full_path.write('Running init {:02d}, fold {:02d} ... \n'.format(init_idx, fold_idx))
            # metrics_file_full_path.write('Running init {:02d}, dataset {} ... \n'.format(init_idx, datasets[dataset_idx]))
            metrics_file_full_path.write('precision= {:.4f}, recall= {:.4f}, f-score= {:.4f}, accuracy= {:.4f} \n\n'.format(precision, recall, fscore, accuracy))
            # print('{:.4f}'.format(sum(labels == label_preds)/labels.shape[0]))
            # common.plot_features(best_features, labels)

    metrics_file_full_path.close()

with open(metrics_file_full_path, 'r') as mf:
    all_lines = mf.readlines()

metrics = np.zeros((num_inits, len(data_train_all_folds), 4))
for line in all_lines:
    splits = line.split(' ')
    if 'init' in line and 'fold' in line:
        init = int(splits[2].split(',')[0])
        fold = int(splits[4])
    elif 'precision' in line and 'recall' in line and 'f-score' in line and 'accuracy' in line:
        metrics[init, fold, 0] = float(splits[1].split(',')[0])
        metrics[init, fold, 1] = float(splits[3].split(',')[0])
        metrics[init, fold, 2] = float(splits[5].split(',')[0])
        metrics[init, fold, 3] = float(splits[7])

stop = 1
