# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 22:07:47 2023

@author: Ying Fu
"""

import platform
import matplotlib


import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Times New Roman'
import torch.nn.init as init

print('--------Packages Versions----------')
print(f'python version: {platform.python_version()}') # 3.9.12
print(f'torch version: {torch.__version__ }')  # 1.12.0
print(f'pandas version: {pd.__version__}')   # 1.4.2
print(f'numpy version: {np.__version__ }')   # 1.23.1
print(f'matplotlib version: {matplotlib.__version__}') # 3.5.1


class LSTM_joint(torch.nn.Module):
    def __init__(self, input_size , window_size, num_layers, hidden1, hidden2, 
                 device):
        super().__init__()
        self.input_size  = input_size   # number of expected features in the input x
        self.window_size = window_size
        self.hidden1 = hidden1  # number of features in the hidden state h
        self.hidden2 = hidden2
        self.num_layers = num_layers    # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results
        self.device = device

        self.lstm0 = torch.nn.LSTM(
            input_size= input_size,
            hidden_size = hidden1,
            num_layers = num_layers,
            batch_first=True,  # the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states.
        )
        
        self.lstm1 = torch.nn.LSTM(
            input_size= input_size,
            hidden_size = hidden1,
            num_layers = num_layers,
            batch_first=True,  # the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states.
        )

        self.fc_branch01 = torch.nn.Linear(in_features = self.hidden1,
                                          out_features = self.hidden2)

        self.fc_branch02 = torch.nn.Linear(in_features = self.hidden2,
                                          out_features = 1)
        
        self.fc_branch11 = torch.nn.Linear(in_features = self.hidden1,
                                          out_features = self.hidden2)

        self.fc_branch12 = torch.nn.Linear(in_features = self.hidden2,
                                          out_features = 1)
        
        self.fm_lin1 = torch.nn.Linear(in_features = input_size*window_size,
                                       out_features = hidden1)
        # # Initialize the weights using Xavier initialization
        # init.xavier_uniform_(self.fm_lin1.weight)
        
        self.fm_lin2 = torch.nn.Linear(in_features=self.hidden1,
                                       out_features=hidden2)
        self.fm_lin3 = torch.nn.Linear(in_features = hidden2,
                                      out_features = 1)
        self.dropout = torch.nn.Dropout(0.2)


    def forward(self, x):
        # print(f"{x.shape = }")  # x:[batch_size, time_window, features]
        batch_size = x.shape[0]
        
        x_flatten = x.flatten(1) # x_flatten:[batch_size, time_window*features]
        # print(f"{x_flatten.shape=}")
        # print(f"{x_flatten = }")
        fm = self.fm_lin1(x_flatten)
        # print(fm)
        # print(f"{fm.shape=}")
        # raise RuntimeError
        fm = F.relu(fm)
        # fm = self.dropout(fm)
        fm = self.fm_lin2(fm)
        fm = F.relu(fm)
        # fm = self.dropout(fm)        
        fm = self.fm_lin3(fm)    
        fm = torch.sigmoid(fm)
        fm = fm.squeeze(1) # (B, 1) -> (B)
        
        h0_0 = torch.zeros(self.num_layers, batch_size, self.hidden1, device = self.device).requires_grad_() # hidden state
        c0_0 = torch.zeros(self.num_layers, batch_size, self.hidden1, device = self.device).requires_grad_() # cell state
        _, (hn_0, cn_0) = self.lstm0(x, (h0_0, c0_0))

        h0_1 = torch.zeros(self.num_layers, batch_size, self.hidden1, device = self.device).requires_grad_() # hidden state
        c0_1 = torch.zeros(self.num_layers, batch_size, self.hidden1, device = self.device).requires_grad_() # cell state
        _, (hn_1, cn_1) = self.lstm1(x, (h0_1, c0_1))
        
        out0 = self.fc_branch01(hn_0[0]) # using the hidden state of the first layer
        out0 = F.relu(out0)
        # out0 = self.dropout(out0)
        out0 = self.fc_branch02(out0)

        out1 = self.fc_branch11(hn_1[0]) # using the hidden state of the first layer
        out1 = F.relu(out1)
        # out1 = self.dropout(out1)
        out1 = self.fc_branch12(out1)

        out0 = out0.squeeze(1)
        out1 = out1.squeeze(1)
        out0_weighted = (1 - fm) * out0
        out1_weighted = fm * out1
        weighted_output = out0_weighted + out1_weighted
        
        return fm, out0, out1, weighted_output
    
    






