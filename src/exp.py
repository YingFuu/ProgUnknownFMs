# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:52:26 2023

@author: Ying Fu
"""

import time
import numpy as np
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random 
import pandas as pd
from matplotlib import pyplot as plt
BIGGER_SIZE = 36

import torch
from torch import optim

from tqdm import tqdm
import gc

# imported packages
import config
from dataloader.sequence_dataloader import SequenceDataset
from dataloader.sequence_grc_dataloader import SequenceDataset_grc
from dataloader.sequence_joint_dataloader import SequenceDataset_joint
from dataloader.sequence_joint_grc_dataloader import SequenceDataset_joint_grc
from models.cnn import DCNN1d
from models.fc import FCNN
from models.lstm import LSTM
from models.lstm_grc import LSTM_grc
from models.lstm_joint_grc import LSTM_joint_grc
from models.lstm_joint import LSTM_joint
from models.branchLSTM import BranchLSTM
from train_evaluate import train_step, val_step, predict_step, \
    train_step_grc, val_step_grc, predict_step_grc, \
    rmse_score, mape_score, health_score, monotonic_ratio, kFold, train_step_joint, val_step_joint, predict_step_joint, \
    train_step_joint_grc, val_step_joint_grc, predict_step_joint_grc


def set_seed(seed):
    '''
    Ensure Reproducibility. Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms.
    Furthermore, results may not be reproducible between CPU and GPU even using same seeds.
    Steps to limit the number of sources of nondeterministic behaviors.
    https://pytorch.org/docs/stable/notes/randomness.html
    '''
    torch.manual_seed(seed) # PyTorch random number genrator.
    np.random.seed(seed) # random number on Numpy
    random.seed(seed) # random seed 
    torch.backends.cudnn.deterministic = True  # using deterministic algorithms
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def HS_loss(output, target):
    '''
    loss = HS(output, target) 
    '''
    d = output - target
    hs_loss = torch.where(d >= 0, torch.exp(d/10 - 1), torch.exp(-d/13 - 1))
    
    return hs_loss.mean()


class HS_GRC_Loss(torch.nn.Module):
    '''
    loss = HS + GRC
    '''
    def __init__(self, lambda0 = 1, lambda1 = 1):
        super(HS_GRC_Loss, self).__init__()
        self.lambda0 = lambda0
        self.lambda1 = lambda1

    def forward(self, output0, target0, output1, target1):
        d = output1 - target1
        hs_loss = torch.where(d >= 0, torch.exp(d/10 - 1), torch.exp(-d/13 - 1))
        pred0_loss = self.lambda0 * (torch.max((output0 - output1) - torch.tensor(1.5/524), torch.tensor(0)))
        pred1_loss = self.lambda1 * (torch.max(torch.tensor(0.5/524) - (output0 - output1), torch.tensor(0)))
        loss = hs_loss + pred0_loss + pred1_loss
        return loss.mean() 


class MSE_GRC_Loss(torch.nn.Module):
    '''
    loss = MSE + GRC
    '''
    def __init__(self, lambda0 = 1, lambda1 = 1):
        super(MSE_GRC_Loss, self).__init__()
        self.lambda0 = lambda0
        self.lambda1 = lambda1

    def forward(self, output0, target0, output1, target1):
        squared_diff = (output1 - target1)**2
        pred0_loss = self.lambda0 * (torch.max((output0 - output1) - torch.tensor(1.5/524), torch.tensor(0)))
        pred1_loss = self.lambda1 * (torch.max(torch.tensor(0.5/524) - (output0 - output1), torch.tensor(0)))
        loss = squared_diff + pred0_loss + pred1_loss
        return loss.mean()  # Return the mean loss across the batch

 

def run(train_loader, val_loader,
        train_step, val_step, 
        model,
        n_epochs, device, loss_fn_regression, optimizer):
    
    # train
    avg_train_loss_lst = []
    avg_val_loss_lst = []
    early_stopper = EarlyStopper(patience=20, min_delta=0.001)
    for epoch in tqdm(range(n_epochs)):
        avg_train_loss = train_step(train_loader, device, model, 
                                    loss_fn_regression, optimizer)
        avg_val_loss = val_step(val_loader, device, model, loss_fn_regression)
        avg_train_loss_lst.append(avg_train_loss)
        avg_val_loss_lst.append(avg_val_loss)
        if early_stopper.early_stop(avg_val_loss):             
            break
        if (epoch + 1) % 20 == 0:
            print(f"epoch = {epoch + 1}, {avg_train_loss:.4f}, {avg_val_loss:.4f}")

    # plot the learning curve
    plt.plot([i for i in range(len(avg_train_loss_lst))], avg_train_loss_lst, label="train")
    plt.plot([i for i in range(len(avg_val_loss_lst))], avg_val_loss_lst, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()




def model_by_model_type(model_type, n_features, cfg):
    '''
    specify the model by the name of model type. 
    the model type should be string.
    '''
    if not isinstance(model_type, str):
        raise RuntimeError
        
    if model_type == 'LSTM': 
        try:
            num_layers = cfg.num_layers
            hidden1 = cfg.hidden1
            hidden2 = cfg.hidden2
        except AttributeError:
            num_layers = 1
            hidden1 = 32
            hidden2 = 32
            cfg.num_layers = num_layers
            cfg.hidden1 = hidden1
            cfg.hidden2 = hidden2
        print(f"{cfg.num_layers =} , {cfg.hidden1 = }, {cfg.hidden2 = }")
        model = LSTM(input_size = n_features, num_layers = cfg.num_layers, 
                     hidden1 = cfg.hidden1, hidden2 = cfg.hidden2,
                     device=cfg.device).to(cfg.device)
    
    elif model_type == 'LSTM-grc':
        try:
            num_layers = cfg.num_layers
            hidden1 = cfg.hidden1
            hidden2 = cfg.hidden2
        except AttributeError:
            num_layers = 1
            hidden1 = 32
            hidden2 = 16
            cfg.num_layers = num_layers
            cfg.hidden1 = hidden1
            cfg.hidden2 = hidden2
        model = LSTM_grc(input_size = n_features, num_layers = cfg.num_layers, 
                     hidden1 = cfg.hidden1, hidden2 = cfg.hidden2,
                     device=cfg.device).to(cfg.device)
    
    elif model_type == 'CNN':        
        try:
            hidden1 = cfg.hidden1
            hidden2 = cfg.hidden2
            kernel_size = cfg.kernel_size
            stride = cfg.stride
        except AttributeError:
            hidden1 = 32
            hidden2 = 32
            kernel_size = 4 
            stride = 1
            cfg.hidden1 = hidden1
            cfg.hidden2 = hidden2
            cfg.kernel_size = kernel_size
            cfg.stride = stride
            
        model = DCNN1d(feature_size=n_features, window_size = cfg.window_size,
                       hidden1=cfg.hidden1, hidden2=cfg.hidden2,
                       kernel_size=cfg.kernel_size, stride=cfg.stride).to(cfg.device)
    
    elif model_type == 'LSTM-joint':
        try:
            num_layers = cfg.num_layers
            hidden1 = cfg.hidden1
            hidden2 = cfg.hidden2
        except AttributeError:
            num_layers = 1
            hidden1 = 32
            hidden2 = 32
            cfg.num_layers = num_layers
            cfg.hidden1 = hidden1
            cfg.hidden2 = hidden2
        model = LSTM_joint(input_size = n_features,
                           window_size = cfg.window_size,
                           num_layers = cfg.num_layers,
                           hidden1 = cfg.hidden1, hidden2 = cfg.hidden2,
                           device=cfg.device).to(cfg.device)
        
    elif model_type == 'BranchLSTM':
        try:
            num_layers = cfg.num_layers
            hidden1 = cfg.hidden1
            hidden2 = cfg.hidden2
        except AttributeError:
            num_layers = 1
            hidden1 = 32
            hidden2 = 32
            cfg.num_layers = num_layers
            cfg.hidden1 = hidden1
            cfg.hidden2 = hidden2
        model = BranchLSTM(input_size = n_features,
                           window_size = cfg.window_size,
                           num_layers = cfg.num_layers,
                           hidden1 = cfg.hidden1, hidden2 = cfg.hidden2,
                           device=cfg.device).to(cfg.device)
   
    elif model_type == 'LSTM-joint-grc':
        try:
            num_layers = cfg.num_layers
            hidden1 = cfg.hidden1
            hidden2 = cfg.hidden2
        except AttributeError:
            num_layers = 1
            hidden1 = 32
            hidden2 = 32
            cfg.num_layers = num_layers
            cfg.hidden1 = hidden1
            cfg.hidden2 = hidden2

        model = LSTM_joint_grc(input_size=n_features,
                               window_size=cfg.window_size,
                               num_layers=cfg.num_layers,
                               hidden1=cfg.hidden1, hidden2=cfg.hidden2,
                               device=cfg.device).to(cfg.device)

    else:
        raise RuntimeError(f'{model_type} is not specified')
        
    return model


def exp(df_train, df_test, model_type,
        SeqDataset,
        train_step_f, val_step_f, predict_step_f,
        no_train_col = ['cycle', 'op1', 'op2', 'op3',
                        'fm_label', 'WC'],
        k_fold_splits = 5,
        loss_fn_regression = torch.nn.MSELoss(reduction='mean'), loss_function_cls = None,
        save_dir = '../result/23-07-17-baseline-FD003-linear'):

    os.makedirs(save_dir, exist_ok=True)    

    rul_max = df_train['RUL'].max()
    rul_min = df_train['RUL'].min()
        
    # initialize the dataset and features    
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    all_columns = list(df_train.columns)
    y_column = ['RUL']
    X_column = list(set(all_columns).difference(set(no_train_col + y_column)))
    norm_col_X = list(set(X_column).difference(set(['id'])))
    n_features = len(norm_col_X)

    train_copy = df_train.copy()
    test_copy = df_test.copy()

    for column in norm_col_X + y_column:
        train_copy[column] = (train_copy[column] - df_train[column].min()) / (df_train[column].max() - df_train[column].min())        
        test_copy[column] = (test_copy[column] - df_train[column].min()) / (df_train[column].max() - df_train[column].min())
    
    test_dataset = SeqDataset(test_copy, no_train_col=no_train_col, 
                                   window_size=cfg.window_size,stride=cfg.stride)
    test_loader = torch.utils.data.DataLoader(test_dataset, cfg.batch_size, shuffle=False)
    
    
    # Perform k-fold cross-validation
    unit_kfold = kFold(df = train_copy, n_splits=k_fold_splits)
    for fold, (train_units_df, val_units_df) in enumerate(unit_kfold.split(train_copy)):
        # train dataset and train loader
        train_dataset = SeqDataset(train_units_df, no_train_col=no_train_col,
                                        window_size=cfg.window_size,stride=cfg.stride)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, 
                                                   shuffle=False)

        val_dataset = SeqDataset(val_units_df, no_train_col=no_train_col,
                                      window_size=cfg.window_size,stride=cfg.stride)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, 
                                                 shuffle=False)

        print(f"{len(train_loader) = }")
        print(f"{len(val_loader) = }")
        

        model = model_by_model_type(model_type, n_features, cfg)
        pytorch_total_params = pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f"{pytorch_total_params = }")
        print(model)
    
        optimizer = optim.Adam(model.parameters(), cfg.lr)

        # training process        
        run(train_loader, val_loader,
            train_step_f, val_step_f,
            model,
            cfg.epochs, cfg.device, loss_fn_regression, optimizer)
        
        # perform prediction on train, val and test dataset
        result_train = predict_step_f(train_loader, cfg.device, model)
        result_val = predict_step_f(val_loader, cfg.device, model)
        result_test = predict_step_f(test_loader, cfg.device, model)
        
        # reverse the normalization
        result_train["pred_rul"] = result_train["pred_rul"].apply(lambda x: x*(rul_max - rul_min) + rul_min)
        result_train["true_rul"] = result_train["true_rul"].apply(lambda x: x*(rul_max - rul_min) + rul_min)
        result_val["pred_rul"] = result_val["pred_rul"].apply(lambda x: x*(rul_max - rul_min) + rul_min)
        result_val["true_rul"] = result_val["true_rul"].apply(lambda x: x*(rul_max - rul_min) + rul_min)
        result_test["pred_rul"] = result_test["pred_rul"].apply(lambda x: x*(rul_max - rul_min) + rul_min)
        result_test["true_rul"] = result_test["true_rul"].apply(lambda x: x*(rul_max - rul_min) + rul_min)

        file_name_train = f'fold-{fold}-result-train.csv'
        file_name_val = f'fold-{fold}-result-var.csv'
        file_name_test = f'fold-{fold}-result-test.csv'

        result_train.to_csv(os.path.join(save_dir, file_name_train), index=False)
        result_val.to_csv(os.path.join(save_dir, file_name_val), index=False)
        result_test.to_csv(os.path.join(save_dir, file_name_test), index=False)
        gc.collect()    
        torch.cuda.empty_cache()




def get_para_lstm_lst(df_train, n_features,
                      hidden1_lst = [16*i for i in range(1,32)],
                      hidden2_lst = [16*i for i in range(1,32)]):
    '''
    Get parameter combination list for validation
    Number of Parameters in LSTM = 4×((input_size+hidden_size)×hidden_size+hidden_size)
    '''
    train_dataset = SequenceDataset(df_train,  
                                    window_size=cfg.window_size,
                                    stride=cfg.stride)
    # # instances
    # print(f"{len(train_dataset)=}")
    para_comb_lst = []    
    for hidden1 in hidden1_lst:
        for hidden2 in hidden2_lst:
            if hidden2 > hidden1:
                continue
            para_lstm = 4*((n_features+hidden1)*hidden1 + hidden1)
            para_ff = hidden1 * hidden2 + hidden2
            para_out = hidden2 * 1 + 1
            total_para_number = para_lstm + para_ff + para_out
            if total_para_number < (2/3*len(train_dataset)): 
                para_comb_lst.append((hidden1,hidden2))
    return para_comb_lst


def exp_all_para(para_comb_lst,
                 df_train, df_test, model_type,
                 SeqDataset,
                 train_step_f, val_step_f, predict_step_f,
                 no_train_col = ['cycle', 'op1', 'op2', 'op3',
                                 'fm_label', 'WC'],
                 k_fold_splits = 5,
                 loss_fn_regression = torch.nn.MSELoss(reduction='mean'),
                 save_all_dir = '../result/23-07-17'):
    
    os.makedirs(save_all_dir, exist_ok=True)
    
    for i, (hidden1,hidden2) in enumerate(para_comb_lst):
        cfg.hidden1 = hidden1
        cfg.hidden2 = hidden2
        exp(df_train, df_test, model_type,
            SeqDataset,
            train_step_f, val_step_f, predict_step_f,
            no_train_col,
            k_fold_splits,
            loss_fn_regression,
            save_dir = f'{save_all_dir}-h1-{hidden1}-h2-{hidden2}')
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()   
        gc.collect()


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def run_exp(model_type = 'LSTM-joint-grc',
            save_dir = '../result-new-FD003', loss = 'mse', lambda0=None):
    
    cfg.lr = 0.0001
    cfg.epochs = 2000
    cfg.window_size= 60
    cfg.batch_size = 128
    set_seed(42) 
    
    save_dir = os.path.join(save_dir, f'{loss}')
    
    ds_name = 'FD003'
    data_dir = f'../result-umap/{ds_name}-unsupervised-linearRUL-neighbors-80-minDist-1-2D'

    df_train  = pd.read_csv(os.path.join(data_dir, 'FD003-train_with_FMlabel.csv'))
    df_test  = pd.read_csv(os.path.join(data_dir, 'FD003-test.csv'))
    df_test['fm_label'] = -1 # -1 means nothing
    
    # perform exp
    no_train_col=['cycle', 'op1', 'op2', 'op3',
                      'fm_label', 'WC']
    all_columns = list(df_train.columns)
    y_column = ['RUL']
    X_column = list(set(all_columns).difference(set(no_train_col + y_column)))
    norm_col_X = list(set(X_column).difference(set(['id'])))
    n_features = len(norm_col_X)
    cfg.num_layers = 1
    
    para_comb_lst = get_para_lstm_lst(df_train, n_features)
    print(f'{para_comb_lst = }')
    
    if model_type in ['LSTM', 'CNN']:
        SeqDataset = SequenceDataset
        train_step_f = train_step
        val_step_f = val_step
        predict_step_f = predict_step
        if loss == 'mse':
            loss_fn_regression=torch.nn.MSELoss(reduction='mean')
        else: 
            loss_fn_regression=HS_loss
        

    elif model_type in ['LSTM-grc']:
        SeqDataset = SequenceDataset_grc
        train_step_f = train_step_grc
        val_step_f = val_step_grc
        predict_step_f = predict_step_grc
        if loss == 'mse':
            loss_fn_regression = MSE_GRC_Loss(lambda0 = lambda0, lambda1 = lambda0)
        else:
            loss_fn_regression = HS_GRC_Loss(lambda0 = lambda0, lambda1 = lambda0)
        save_dir = save_dir + f'-lambda-{lambda0}'
        
    elif model_type in ['LSTM-joint', 'BranchLSTM']:
        SeqDataset = SequenceDataset_joint
        train_step_f = train_step_joint
        val_step_f = val_step_joint
        predict_step_f = predict_step_joint
        if loss == 'mse':
            loss_fn_regression = torch.nn.MSELoss(reduction='mean')
        else:
            loss_fn_regression=HS_loss


    elif model_type in ['LSTM-joint-grc']:
        SeqDataset = SequenceDataset_joint_grc
        train_step_f = train_step_joint_grc
        val_step_f = val_step_joint_grc
        predict_step_f = predict_step_joint_grc
        if loss == 'mse':
            loss_fn_regression = MSE_GRC_Loss(lambda0 = lambda0, lambda1 = lambda0)
        else:
            loss_fn_regression = HS_GRC_Loss(lambda0 = lambda0, lambda1 = lambda0)
        save_dir = save_dir + f'-lambda-{lambda0}'
    
    else:
        raise RuntimeError
        
    exp_all_para(para_comb_lst=para_comb_lst,
                 df_train=df_train, df_test=df_test, model_type=model_type,
                 SeqDataset = SeqDataset,
                 train_step_f = train_step_f, val_step_f=val_step_f, predict_step_f=predict_step_f,
                 k_fold_splits=5,
                 loss_fn_regression=loss_fn_regression,
                 save_all_dir=os.path.join(save_dir, model_type))

    
if __name__ == "__main__":
    
    cfg = config.Config_basic()
    
    for model_type in ['LSTM', 'CNN']:
        run_exp(model_type = model_type,
                save_dir = f'../result-FD003-{model_type}',
                loss = 'hs', lambda0=None)

    
    for model_type in ['LSTM-joint']:
        run_exp(model_type = model_type,
                save_dir = f'../result-FD003-{model_type}',
                loss = 'hs', lambda0=None)


    for model_type in ['BranchLSTM']:
        run_exp(model_type = model_type,
                save_dir = f'../result-FD003-{model_type}',
                loss = 'mse', lambda0=None)
        
        
    for lambda0 in [0.1,0.5,1,2,4]:
        for model_type in ['LSTM-joint-grc']:
            run_exp(model_type = model_type,
                    save_dir = f'../result-FD003-{model_type}-{lambda0}',
                    loss = 'hs', lambda0=lambda0)
    
    






