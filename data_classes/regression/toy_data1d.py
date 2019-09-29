import os
import re

import numpy as np
import torch
import torch.utils.data as data


class ToyData1D(data.Dataset):
    def __init__(self,
                 root,
                 data_set=None,
                 train=True,
                 x_points=[],
                 base=None,
                 noise=None):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.base_model = base
        self.noise_model = noise

        if self.train:
            self.train_data, self.train_labels = torch.FloatTensor(
                x_points[0]), torch.FloatTensor(
                    self.sample_data(x_points[0]).reshape(-1))
        else:
            self.test_data, self.test_labels = torch.FloatTensor(
                x_points[1]), torch.FloatTensor(
                    self.sample_data(x_points[1]).reshape(-1))

    def sample_data(self, x):
        return self.base_model(x) + np.random.normal(0, self.noise_model(x))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            x, y = self.train_data[index], self.train_labels[index]
        else:
            x, y = self.test_data[index], self.test_labels[index]

        return x, y

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
