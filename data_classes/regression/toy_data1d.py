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
                 y_points=[]):
        #  base=None,
        #  noise=None):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        if self.train:
            self.train_data = torch.FloatTensor(x_points[0]).view(-1, 1)
            self.train_labels = torch.FloatTensor(y_points[0]).view(-1)
        else:
            self.test_data = torch.FloatTensor(x_points[1]).view(-1, 1)
            self.test_labels = torch.FloatTensor(y_points[1]).view(-1)

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
