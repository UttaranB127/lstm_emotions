import argparse
import os
import numpy as np
from utils import loader, processor, common

import torch
from torchlight.torchlight.gpu import ngpu


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

data, labels, data_train_all_folds, labels_train_all_folds,\
    data_test_all_folds, labels_test_all_folds = \
        loader.load_data(data_path, ftype, joints, coords, cycles=cycles, num_folds=num_folds)
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

        best_features, label_preds = pr.extract_best_feature(data, joints, coords)
        # print('{:.4f}'.format(sum(labels == label_preds)/labels.shape[0]))
        # common.plot_features(best_features, labels)

        # TO DO: calculate and save precision, recall, and accuracy for each init and fold
