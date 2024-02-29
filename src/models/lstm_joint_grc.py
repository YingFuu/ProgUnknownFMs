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


class LSTM_joint_grc(torch.nn.Module):
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
        
        self.fm_lin1 = torch.nn.Linear(in_features = input_size*(window_size-1),
                                       out_features = hidden1)
        # # Initialize the weights using Xavier initialization
        # init.xavier_uniform_(self.fm_lin1.weight)
        
        self.fm_lin2 = torch.nn.Linear(in_features=self.hidden1,
                                       out_features=hidden2)
        self.fm_lin3 = torch.nn.Linear(in_features = hidden2,
                                      out_features = 1)
        # self.dropout = torch.nn.Dropout(0.2)


    def forward(self, x0, x1):
        # x0: [0:window_size-1]
        # x1: [1:window_size]

        batch_size = x1.shape[0]

        x_flatten_0 = x0.flatten(1) # x_flatten:[batch_size, time_window*features]
        fm0 = self.fm_lin1(x_flatten_0)
        fm0 = F.relu(fm0)
        fm0 = self.fm_lin2(fm0)
        fm0 = F.relu(fm0)
        fm0 = self.fm_lin3(fm0)
        fm0 = torch.sigmoid(fm0)
        fm0 = fm0.squeeze(1) # (B, 1) -> (B)

        x_flatten1 = x1.flatten(1) # x_flatten:[batch_size, time_window*features]
        fm1 = self.fm_lin1(x_flatten1)
        fm1 = F.relu(fm1)
        fm1 = self.fm_lin2(fm1)
        fm1 = F.relu(fm1)
        fm1 = self.fm_lin3(fm1)
        fm1 = torch.sigmoid(fm1)
        fm1 = fm1.squeeze(1) # (B, 1) -> (B)

        
        h0_0 = torch.zeros(self.num_layers, batch_size, self.hidden1, device = self.device).requires_grad_() # hidden state
        c0_0 = torch.zeros(self.num_layers, batch_size, self.hidden1, device = self.device).requires_grad_() # cell state
        _, (hn_00, cn_00) = self.lstm0(x0, (h0_0, c0_0)) # 0th failure mode lstm layer, 0th output
        _, (hn_01, cn_01) = self.lstm0(x1, (h0_0, c0_0)) # 0th failure mode lstm layer, 1th output


        h0_1 = torch.zeros(self.num_layers, batch_size, self.hidden1, device = self.device).requires_grad_() # hidden state
        c0_1 = torch.zeros(self.num_layers, batch_size, self.hidden1, device = self.device).requires_grad_() # cell state
        _, (hn_10, cn_10) = self.lstm1(x0, (h0_1, c0_1)) # 1th failure mode lstm layer, 0th output
        _, (hn_11, cn_11) = self.lstm1(x1, (h0_1, c0_1)) # 1th failure mode lstm layer, 1th output

        out00 = self.fc_branch01(hn_00[0])
        out00 = F.relu(out00)
        # out00 = self.dropout(out00)
        out00 = self.fc_branch02(out00)

        out01 = self.fc_branch01(hn_01[0])
        out01 = F.relu(out01)
        # out01 = self.dropout(out01)
        out01 = self.fc_branch02(out01)


        out10 = self.fc_branch11(hn_10[0])
        out10 = F.relu(out10)
        # out10 = self.dropout(out10)
        out10 = self.fc_branch12(out10)

        out11 = self.fc_branch11(hn_11[0])
        out11 = F.relu(out11)
        # out11 = self.dropout(out11)
        out11 = self.fc_branch12(out11)


        out00 = out00.squeeze(1) # 0th failure mode lstm layer, 0th output
        out01 = out01.squeeze(1) # 0th failure mode lstm layer, 1st output

        out10 = out10.squeeze(1) # 1th failure mode lstm layer, 0th output
        out11 = out11.squeeze(1) # 1th failure mode lstm layer, 1st output

        return fm0, fm1, (1-fm0)*out00 + fm0*out10, (1-fm1)*out01 + fm1*out11
        






