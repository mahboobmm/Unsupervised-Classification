"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
import sys
import numpy as np
import torch

import cv2 as cv

from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from utils.mypath import MyPath
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class CustomDataset(Dataset):
    """
    Args:
        root (string): Root directory of custom dataset where ``custom-dataset-images``
            exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    base_folder = 'custom-dataset-images'

    def __init__(self, root=MyPath.db_root_dir('custom-dataset'), train=True, transform=None, 
                    download=False):

        super(CustomDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set

        self.data = []
        self.targets = []

        for file_path in glob(os.path.join(self.root, self.base_folder, '*')):
            entry = cv.cvtColor(cv.imread(file_path), cv.color_BGR2RGB)
            self.data.append(entry)

        self.target = np.zeros(self.__len__())

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index}}
        
        return out

    def get_image(self, index):
        img = self.data[index]
        return img
        
    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")