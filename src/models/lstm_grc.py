# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 21:34:02 2022

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


print('--------Packages Versions----------')
print(f'python version: {platform.python_version()}') # 3.9.12
print(f'torch version: {torch.__version__ }')  # 1.12.0
print(f'pandas version: {pd.__version__}')   # 1.4.2
print(f'numpy version: {np.__version__ }')   # 1.23.1
print(f'matplotlib version: {matplotlib.__version__}') # 3.5.1


class LSTM_grc(torch.nn.Module):
    def __init__(self, input_size , num_layers, hidden1, hidden2,
                 device):
        super().__init__()
        self.input_size  = input_size   # number of expected features in the input x
        self.hidden1 = hidden1  # number of features in the hidden state h
        self.hidden2 = hidden2  # number of features in feed forward neural network
        self.num_layers = num_layers    # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results
        self.device = device

        self.lstm = torch.nn.LSTM(
                    input_size= input_size,
                    hidden_size = hidden1,
                    num_layers = num_layers,
                    batch_first=True,  # the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states.
        )
        
        self.fc1 = torch.nn.Linear(in_features = self.hidden1,
                                      out_features = self.hidden2)
        self.fc2 = torch.nn.Linear(in_features = self.hidden2,
                                      out_features = 1)
        self.dropout = torch.nn.Dropout(0.2)


    def forward(self, x0, x1):
        batch_size = x0.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden1, device = self.device).requires_grad_() # hidden state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden1, device = self.device).requires_grad_() # cell state
        
        # output, (hn, cn) = self.lstm(x, (h0, c0))
        # output: (batch_size, seq_len, hidden_size): 
        #           output[i,t]: the hidden state in last layer for t-th element
        #                          in i-th sequence of a batch.
        # hn, cn: (num_layers, batch_size, hidden_size): 
        #           hn[l,i]: the hidden state in i-th layer for last element
        #                          in i-th sequence of a batch.
        _ , (hn_0, cn_0) = self.lstm(x0, (h0, c0))
        out_0 = self.fc1(hn_0[-1]) # using the hidden state in the last layer of at last time step
        out_0 = F.relu(out_0)
        # out_0 = self.dropout(out_0)
        out_0 = self.fc2(out_0)
        out_0 = out_0.squeeze(1) # turn [B,1] to [B]

        _ , (hn_1, cn_1) = self.lstm(x1, (h0, c0))
        out_1 = self.fc1(hn_1[-1]) # using the hidden state in the last layer of at last time step
        out_1 = F.relu(out_1)
        # out_1 = self.dropout(out_1)
        out_1 = self.fc2(out_1)
        out_1 = out_1.squeeze(1) # turn [B,1] to [B]
        
        return out_0,out_1








