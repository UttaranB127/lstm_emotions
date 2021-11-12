# sys
import h5py
import math
import os
import numpy as np
# import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold

# torch
import torch
from torchvision import datasets, transforms

from utils import common


def load_data(_path, _ftype, joints, coords, cycles=3, num_folds=10, test_size=0.1):
    """
    test_size has no effect when n_folds is used
    """

    file_affeature = os.path.join(_path, 'affectiveFeatures.h5')
    ffa = h5py.File(file_affeature, 'r')

    file_gait = os.path.join(_path, 'features' + _ftype + '.h5')
    ffg = h5py.File(file_gait, 'r')

    file_label = os.path.join(_path, 'labels' + _ftype + '.h5')
    fl = h5py.File(file_label, 'r')

    aff_list = []
    num_samples = len(ffa.keys())
    for si in range(num_samples):
        ffa_group_key = list(ffa.keys())[si]
        aff_list.append(ffa[ffa_group_key])
    aff = np.array(aff_list)

    gait_list = []
    num_samples = len(ffg.keys())
    time_steps = 0
    labels = np.empty(num_samples)
    for si in range(num_samples):
        ffg_group_key = list(ffg.keys())[si]
        gait_list.append(list(ffg[ffg_group_key]))  # Get the data
        time_steps_curr = len(ffg[ffg_group_key])
        if time_steps_curr > time_steps:
            time_steps = time_steps_curr
        labels[si] = fl[list(fl.keys())[si]][()]

    gait = np.empty((num_samples, time_steps*cycles, joints*coords))
    for si in range(num_samples):
        data_list_curr = np.tile(gait_list[si], (int(np.ceil(time_steps / len(gait_list[si]))), 1))
        for ci in range(cycles):
            gait[si, time_steps * ci:time_steps * (ci + 1), :] = data_list_curr[0:time_steps]
    # gait = common.get_affective_features(np.reshape(gait, (gait.shape[0], gait.shape[1], joints, coords)))[:, :, :48]

    data = [(a, g) for a, g in zip(aff, gait)]
    data_train = [None] * num_folds
    labels_train = [None] * num_folds
    data_test = [None] * num_folds
    labels_test = [None] * num_folds
    kf = KFold(n_splits=num_folds, shuffle=True)
    for idx, (train_indices, test_indices) in enumerate(kf.split(data)):
        data_train[idx] = [data[tr] for tr in train_indices]
        labels_train[idx] = [labels[tr] for tr in train_indices]
        data_test[idx] = [data[ts] for ts in test_indices]
        labels_test[idx] = [labels[ts] for ts in test_indices]
    return data, labels, data_train, labels_train, data_test, labels_test
    # data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size)
    # return data, labels, data_train, labels_train, data_test, labels_test


def scale(_data):
    data_scaled = _data.astype('float32')
    data_max = np.max(data_scaled)
    data_min = np.min(data_scaled)
    data_scaled = (_data-data_min)/(data_max-data_min)
    return data_scaled, data_max, data_min


# descale generated data
def descale(data, data_max, data_min):
    data_descaled = data*(data_max-data_min)+data_min
    return data_descaled


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


class TrainTestLoader(torch.utils.data.Dataset):

    def __init__(self, data, label, joints, coords):
        # load data
        self.aff = []
        self.gait = []
        for aff, gait in data:
            self.aff.append(aff)
            self.gait.append(gait)
        self.aff = np.array(self.aff)
        self.gait = np.array(self.gait)
        self.gait = np.reshape(self.gait, (self.gait.shape[0], self.gait.shape[1], joints, coords, 1))
        self.gait = np.moveaxis(self.gait, [1, 2, 3], [2, 3, 1])

        # load label
        self.label = label

        self.F = self.aff.shape[1]
        self.N, self.C, self.T, self.J, self.M = self.gait.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        aff = np.array(self.aff[index])
        gait = np.array(self.gait[index])
        label = self.label[index]
        return aff, gait, label
