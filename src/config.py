# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 12:04:38 2022

@author: Ying Fu
"""

import torch


class Config_basic():
    def __init__(self):
        # Data
        self.ds_dir = '../dataset/Aircraft Engine/CMaps'
        self.train_file_name = 'train_FD003.txt'
        self.test_file_name = 'test_FD003.txt'
        self.gt_file_name = 'RUL_FD003.txt'
        self.normalizer = None

        # pytorch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # training
        self.lr = 0.01
        self.epochs = 250
        self.batch_size = 256

        self.window_size = 30
        self.stride = 1
        self.shuffle = False
        self.dropout = 0


