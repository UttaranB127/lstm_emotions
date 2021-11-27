import glob
import h5py
import math
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from net import classifier
from torchlight.torchlight.io import IO


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def find_all_substr(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_best_epoch_and_accuracy(path_to_model_files):
    all_models = [name.split('/')[-1] for name in glob.glob(os.path.join(path_to_model_files, '*_model.pth.tar'))]
    if len(all_models) == 0:
        return None, 0.
    acc_list = np.zeros(len(all_models))
    for i, model in enumerate(all_models):
        acc_list[i] = float(model.split('_')[3])
    best_model = all_models[np.argmax(acc_list)].split('_')
    return int(best_model[1]), float(best_model[3])


class Processor(object):
    """
        Processor for gait generation
    """

    def __init__(self, args, data_loader, C, F, num_classes, graph_dict, device='cuda:0'):

        self.args = args
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.device = device
        self.io = IO(
            self.args.work_dir,
            save_log=self.args.save_log,
            print_log=self.args.print_log)

        # model
        self.model = classifier.Classifier(C, F, num_classes, graph_dict)
        self.model.cuda('cuda:0')
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        self.best_loss = math.inf
        self.step_epochs = [math.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
        self.best_epoch = None
        self.best_accuracy = np.zeros((1, np.max(self.args.topk)))
        self.accuracy_updated = False

        # optimizer
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        self.lr = self.args.base_lr

    def adjust_lr(self):

        # if self.args.optimizer == 'SGD' and \
        if self.meta_info['epoch'] in self.step_epochs:
            lr = self.args.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.step_epochs)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def show_epoch_info(self):

        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.args.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):

        if self.meta_info['iter'] % self.args.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.args.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def show_topk(self, k, epoch):

        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100. * sum(hit_top_k) * 1.0 / len(hit_top_k)
        if accuracy > self.best_accuracy[0, k-1]:
            self.best_accuracy[0, k-1] = accuracy
            self.accuracy_updated = True
            self.best_epoch = epoch
        else:
            self.accuracy_updated = False
        print_epoch = self.best_epoch if self.best_epoch is not None else 0
        self.io.print_log('\tTop{}: {:.2f}%. Best so far: {:.2f}% (epoch: {:d}).'.
                          format(k, accuracy, self.best_accuracy[0, k - 1], print_epoch))

    def per_train(self):

        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for aff, gait, label in loader:
            # get data
            aff = aff.float().to(self.device)
            gait = gait.float().to(self.device)
            label = label.long().to(self.device)

            # forward
            output, _ = self.model(aff, gait)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def per_test(self, epoch=None, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for aff, gait, label in loader:

            # get data
            aff = aff.float().to(self.device)
            gait = gait.float().to(self.device)
            label = label.long().to(self.device)

            # inference
            with torch.no_grad():
                output, _ = self.model(aff, gait)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.args.topk:
                self.show_topk(k, epoch)

    def train(self):

        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            self.meta_info['epoch'] = epoch

            # training
            self.io.print_log('Training epoch: {}'.format(epoch))
            self.per_train()
            self.io.print_log('Done.')

            # evaluation
            if (epoch % self.args.eval_interval == 0) or (
                    epoch + 1 == self.args.num_epoch):
                self.io.print_log('Eval epoch: {}'.format(epoch))
                self.per_test(epoch=epoch)
                self.io.print_log('Done.')

            # save model and weights
            if self.accuracy_updated:
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.work_dir,
                                        'epoch_{}_acc_{:.2f}_model.pth.tar'.format(epoch, self.best_accuracy.item())))

    def test(self):

        # the path of weights must be appointed
        if self.args.weights is None:
            raise ValueError('Please appoint --weights.')
        self.io.print_log('Model:   {}.'.format(self.args.model))
        self.io.print_log('Weights: {}.'.format(self.args.weights))

        # evaluation
        self.io.print_log('Evaluation Start:')
        self.per_test()
        self.io.print_log('Done.\n')

        # save the output of model
        if self.args.save_result:
            result_dict = dict(
                zip(self.data_loader['test'].dataset.sample_name,
                    self.result))
            self.io.save_pkl(result_dict, 'test_result.pkl')

    def extract_best_feature(self, data, joints, coords):
        if self.best_epoch is None:
            self.best_epoch, best_accuracy = get_best_epoch_and_accuracy(self.args.work_dir)
        else:
            best_accuracy = self.best_accuracy.item()
        if self.best_epoch is not None:
            filename = os.path.join(self.args.work_dir,
                                    'epoch_{}_acc_{:.2f}_model.pth.tar'.format(self.best_epoch, best_accuracy))
            self.model.load_state_dict(torch.load(filename))
        label_preds = np.empty(len(data))
        features = np.empty((0, 4))

        for i, each_data in enumerate(data):
            # get data
            aff = each_data[0]
            aff = torch.from_numpy(aff).float().to(self.device)
            aff = aff.unsqueeze(0)
            gait = each_data[1]
            gait = np.reshape(gait, (1, gait.shape[0], joints, coords, 1))
            gait = np.moveaxis(gait, [1, 2, 3], [2, 3, 1])
            gait = torch.from_numpy(gait).float().to(self.device)

            # get feature
            self.model.eval()
            with torch.no_grad():
                output, feature = self.model(aff, gait)
                label_preds[i] = np.argmax(output.cpu().numpy())
                features = np.append(features, feature.cpu().numpy().reshape((1, feature.shape[1])), axis=0)

        return features, label_preds
