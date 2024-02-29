# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 22:07:47 2023

Implementation of paper:
    Li, Zhen, et al. "A deep branched network for failure mode diagnostics and remaining useful life prediction." IEEE Transactions on Instrumentation and Measurement 71 (2022): 1-11.

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


class BranchLSTM(torch.nn.Module):
    def __init__(self, input_size , window_size, num_layers, hidden1, hidden2, 
                 device):
        super().__init__()
        self.input_size  = input_size   # number of expected features in the input x
        self.window_size = window_size
        self.hidden1 = hidden1  # number of features in the hidden state h
        self.hidden2 = hidden2
        self.num_layers = num_layers    # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results
        self.device = device
        
        # Shared Representation Layers
        self.lstm = torch.nn.LSTM(
            input_size= input_size,
            hidden_size = hidden1,
            num_layers = num_layers,
            batch_first=True,  # the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states.
        )
        self.fc_shared = torch.nn.Linear(in_features=self.hidden1, out_features=8)
        
        # Classification Layers
        self.fm_lin1 = torch.nn.Linear(in_features=8,
                                       out_features=self.hidden2)
        self.fm_lin2 = torch.nn.Linear(in_features = self.hidden2,
                                      out_features = 1)

        # Regression Layers for Mode 0
        self.fc_branch01 = torch.nn.Linear(in_features = 8,
                                          out_features = self.hidden2)

        self.fc_branch02 = torch.nn.Linear(in_features = self.hidden2,
                                          out_features = 1)
        
        # Regression Layers for Mode 1
        self.fc_branch11 = torch.nn.Linear(in_features = 8,
                                          out_features = self.hidden2)

        self.fc_branch12 = torch.nn.Linear(in_features = self.hidden2,
                                          out_features = 1)
                
        self.dropout = torch.nn.Dropout(0.2)


    def forward(self, x):
        # print(f"{x.shape = }")  # x:[batch_size, time_window, features]
        batch_size = x.shape[0]
        
        # # Feature Extraction
        # 1. LSTM layer
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden1, device = self.device).requires_grad_() # hidden state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden1, device = self.device).requires_grad_() # cell state
        _, (hn, cn) = self.lstm(x, (h0, c0))
        # 2. FC layer
        out_common = self.fc_shared(hn[0]) # using the hidden state of the first layer
        out_common = F.relu(out_common)
        
        # # Classification Layers
        fm = self.fm_lin1(out_common)
        fm = F.relu(fm)
        fm = self.fm_lin2(fm)
        fm = torch.sigmoid(fm)
        fm = fm.squeeze(1) # (B, 1) -> (B)
        
        # # FM0
        out0 = self.fc_branch01(out_common) # using the hidden state of the first layer
        out0 = F.relu(out0)
        out0 = self.fc_branch02(out0)
        
        # # FM1
        out1 = self.fc_branch11(out_common) # using the hidden state of the first layer
        out1 = F.relu(out1)
        out1 = self.fc_branch12(out1)

        out0 = out0.squeeze(1)
        out1 = out1.squeeze(1)
        out0_weighted = (1 - fm) * out0
        out1_weighted = fm * out1
        weighted_output = out0_weighted + out1_weighted
        
        return fm, out0, out1, weighted_output
    
    






