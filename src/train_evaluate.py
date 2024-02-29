# -*- coding: utf-8 -*-
"""
Created on Sun May 28 16:11:45 2023

@author: Ying Fu
"""
import torch
torch.manual_seed(42)
import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.model_selection import KFold

def tensor_2_numpy(tensor_v):
    '''
    convert a tensor vector to a numpy vector
    transferring data between GPU and CPU may be costly
    # deatch --> cut computational graph
    # cpu --> allocate tensor in RAM
    # numpy --> port tensor to numpy
    '''
    if type(tensor_v) is np.ndarray:
        return tensor_v
    if tensor_v.is_cuda: # stored on the gpu
        tensor_v = tensor_v.cpu().detach().numpy() # gpu-->cpu-->numpy
    else:  # stored on the cpu
        tensor_v = tensor_v.detach().numpy() # cpu --> numpy
    return tensor_v

####### train_step, val_step, predict_model of baseline model #######
def train_step(train_loader, device, model, loss_function, optimizer):
    '''
    train a model
    :param train_loader: DataLoader class
    :param device: run on cpu or CUDA
    :param model: a specified model, LSTM or CNN, or any custom models.
    :param loss_function:
    :param optimizer:
    :return: float, average train loss
    '''
    num_batches = len(train_loader)

    total_loss = 0
    model.train()
    for X, y, _ in train_loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(X).to(device)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def val_step(val_loader, device, model, loss_function):
    '''
    evaluate a model
    :param val_loader: DataLoader class
    :param device: run on cpu or CUDA
    :param model: already trained model
    :param loss_function:
    :return: float, average validation loss
    '''
    num_batches = len(val_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():  # will not use CUDA memory
        for X, y, _ in val_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X).to(device)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches

    return avg_loss


def predict_step(test_loader, device, model):
    '''
    predict the model
    :param test_loader: DataLoader class
    :param model: already trained model.
    :param device: run on cpu or CUDA
    :return: prediction and true, torch.tensor
    '''
    num_batches = len(test_loader)
    print(f"{num_batches = }")
    batch_size = test_loader.batch_size

    num_elements = len(test_loader.dataset)
    
    result = torch.zeros((num_elements,3))
    model.eval()
    with torch.no_grad():
        for i, (X, y, data_id) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)
            data_id = data_id.to(device)
            start = i*batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements
            y_pred= model(X)
            
            result[start:end, 0] = data_id
            result[start:end, 1] = y_pred
            result[start:end, 2] = y
        
    result = tensor_2_numpy(result)        
    result = pd.DataFrame(result, columns = ['unit_id',
                                             'pred_rul',
                                             'true_rul'])
    return result



####### train_step, val_step, predict_model of baseline GRC(Growth Rate Constrainted)model #######
def train_step_grc(train_loader, device, model, loss_function, optimizer):
    '''
    train a model
    :param train_loader: DataLoader class
    :param device: run on cpu or CUDA
    :param model: a specified model, LSTM or CNN, or any custom models.
    :param loss_function:
    :param optimizer:
    :return: float, average train loss
    '''
    num_batches = len(train_loader)

    total_loss = 0
    model.train()
    for X0, y0, X1, y1, _ in train_loader:
        X0 = X0.to(device)
        y0 = y0.to(device)
        X1 = X1.to(device)
        y1 = y1.to(device)
        optimizer.zero_grad()
        out0, out1 = model(X0, X1)
        out0 = out0.to(device)
        out1 = out1.to(device)
        loss = loss_function(out0, y0, out1, y1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def val_step_grc(val_loader, device, model, loss_function):
    '''
    evaluate a model
    :param val_loader: DataLoader class
    :param device: run on cpu or CUDA
    :param model: already trained model
    :param loss_function:
    :return: float, average validation loss
    '''
    num_batches = len(val_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():  # will not use CUDA memory
        for X0, y0, X1, y1, _ in val_loader:
            X0 = X0.to(device)
            y0 = y0.to(device)
            X1 = X1.to(device)
            y1 = y1.to(device)
            out0, out1 = model(X0, X1)
            out0 = out0.to(device)
            out1 = out1.to(device)
            total_loss += loss_function(out0, y0, out1, y1).item()

    avg_loss = total_loss / num_batches

    return avg_loss


def predict_step_grc(test_loader, device, model):
    '''
    predict the model
    :param test_loader: DataLoader class
    :param model: already trained model.
    :param device: run on cpu or CUDA
    :return: prediction and true, torch.tensor
    '''
    num_batches = len(test_loader)
    print(f"{num_batches = }")
    batch_size = test_loader.batch_size

    num_elements = len(test_loader.dataset)
    
    result = torch.zeros((num_elements,3))
    model.eval()
    with torch.no_grad():
        for i, (X0, _, X1, y1, data_id) in enumerate(test_loader):
            X0 = X0.to(device)
            # y0 = y0.to(device)
            X1 = X1.to(device)
            y1 = y1.to(device)        
            data_id = data_id.to(device)
            start = i*batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements
            y_pred0, y_pred1= model(X0,X1)
            
            result[start:end, 0] = data_id
            result[start:end, 1] = y_pred1
            result[start:end, 2] = y1
        
    result = tensor_2_numpy(result)        
    result = pd.DataFrame(result, columns = ['unit_id',
                                             'pred_rul',
                                             'true_rul'])
    return result


####### train_step, val_step, predict_model of joint model #######
def train_step_joint(train_loader, device, model,
                     loss_function_reg,
                     optimizer):
    '''
    train a model
    :param train_loader: DataLoader class
    :param device: run on cpu or CUDA
    :param model: a specified model, LSTM or CNN, or any custom models.
    :param loss_function:
    :param optimizer:
    :return: float, average train loss

    Parameters
    ----------
    loss_function_cls
    loss_function_reg
    '''
    loss_function_cls = torch.nn.BCELoss(reduction='mean')
    num_batches = len(train_loader)

    total_loss = 0
    model.train()
    for X, y, _, fm_label in train_loader:
        X = X.to(device)
        y = y.to(device)
        fm = fm_label.to(device)
        optimizer.zero_grad()
        fm_p, _, _, rul_p = model(X)
        fm_p = fm_p.to(device)
        rul_p = rul_p.to(device)
        loss_rul = loss_function_reg(rul_p, y)
        loss_fm = loss_function_cls(fm_p, fm)
        # print(f"{loss_rul = }")
        # print(f"{loss_fm = }")
        loss = 10*loss_rul + loss_fm  # loss_fm may be two times of loss_rul
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def val_step_joint(val_loader, device, model, loss_function_reg):
    '''
    evaluate a model
    :param val_loader: DataLoader class
    :param device: run on cpu or CUDA
    :param model: already trained model
    :param loss_function:
    :return: float, average validation loss
    '''
    loss_function_cls = torch.nn.BCELoss(reduction='mean')
    num_batches = len(val_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():  # will not use CUDA memory
        for X, y, _, fm_label in val_loader:
            X = X.to(device)
            y = y.to(device)
            fm = fm_label.to(device)
            fm_p, _, _, rul_p = model(X)
            fm_p = fm_p.to(device)
            rul_p = rul_p.to(device)
            loss_rul = loss_function_reg(rul_p, y)
            loss_fm = loss_function_cls(fm_p, fm)
            # print(f"{loss_rul = }")   
            # print(f"{loss_fm = }")
            loss = 10*loss_rul + loss_fm  # loss_fm may be two times of loss_rul
            total_loss += loss.item()
    avg_loss = total_loss / num_batches

    return avg_loss


def predict_step_joint(test_loader, device, model):
    '''
    predict the model
    :param test_loader: DataLoader class
    :param model: already trained model.
    :param device: run on cpu or CUDA
    :return: prediction and true, torch.tensor
    '''

    num_batches = len(test_loader)
    batch_size = test_loader.batch_size

    num_elements = len(test_loader.dataset)

    result = torch.zeros((num_elements, 5))
    model.eval()
    with torch.no_grad():
        for i, (X, y, data_id, fm_label) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)
            fm = fm_label.to(device)
            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements
    
            fm_p, rul_0, rul_1, rul_p = model(X)
    
            result[start:end, 0] = data_id
            result[start:end, 1] = fm_p
            result[start:end, 2] = fm
            result[start:end, 3] = rul_p
            result[start:end, 4] = y

    result = tensor_2_numpy(result)
    result = pd.DataFrame(result, columns=['unit_id',
                                           'pred_fm',
                                           'fm',
                                           'pred_rul',
                                           'true_rul'])
    return result


####### train_step, val_step, predict_model of joint + GRC(Growth Rate Constrainted)model #######
def train_step_joint_grc(train_loader, device, model,
                        loss_function_reg,
                        optimizer):
    '''
    train a model
    :param train_loader: DataLoader class
    :param device: run on cpu or CUDA
    :param model: a specified model, LSTM or CNN, or any custom models.
    :param loss_function:
    :param optimizer:
    :return: float, average train loss

    Parameters
    ----------
    loss_function_cls
    loss_function_reg
    '''
    loss_function_cls = torch.nn.BCELoss(reduction='mean')
    num_batches = len(train_loader)

    total_loss = 0
    model.train()
    for X0, y0, X1, y1, _, fm_label in train_loader:
        X0 = X0.to(device)
        y0 = y0.to(device)
        X1 = X1.to(device)
        y1 = y1.to(device)
        fm = fm_label.to(device)

        optimizer.zero_grad()
        _, fm_p1, rul_p0, rul_p1 = model(X0, X1)
        fm_p1 = fm_p1.to(device)
        rul_p0 = rul_p0.to(device)
        rul_p1 = rul_p1.to(device)

        loss_rul = loss_function_reg(rul_p0, y0, rul_p1, y1)
        loss_fm = loss_function_cls(fm_p1, fm)
        loss = 10 * loss_rul + loss_fm
        loss.backward()
        # clip the gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def val_step_joint_grc(val_loader, device, model, loss_function_reg):
    '''
    evaluate a model
    :param val_loader: DataLoader class
    :param device: run on cpu or CUDA
    :param model: already trained model
    :param loss_function:
    :return: float, average validation loss
    '''
    loss_function_cls = torch.nn.BCELoss(reduction='mean')
    num_batches = len(val_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():  # will not use CUDA memory
        for X0, y0, X1, y1, _, fm_label in val_loader:
            X0 = X0.to(device)
            y0 = y0.to(device)
            X1 = X1.to(device)
            y1 = y1.to(device)
            fm = fm_label.to(device)
            _, fm_p1, rul_p0, rul_p1 = model(X0, X1)
            fm_p1 = fm_p1.to(device)
            rul_p0 = rul_p0.to(device)
            rul_p1 = rul_p1.to(device)
            loss_rul = loss_function_reg(rul_p0, y0, rul_p1, y1)
            loss_fm = loss_function_cls(fm_p1, fm)
            loss = 10 * loss_rul + loss_fm
            total_loss += loss.item()
    avg_loss = total_loss / num_batches

    return avg_loss


def predict_step_joint_grc(test_loader, device, model):
    '''
    predict the model
    :param test_loader: DataLoader class
    :param model: already trained model.
    :param device: run on cpu or CUDA
    :return: prediction and true, torch.tensor
    '''

    num_batches = len(test_loader)
    batch_size = test_loader.batch_size

    num_elements = len(test_loader.dataset)

    result = torch.zeros((num_elements, 5))
    model.eval()
    with torch.no_grad():
        for i, (X0, y0, X1, y1, data_id, fm_label) in enumerate(test_loader):
            X0 = X0.to(device)
            y0 = y0.to(device)
            X1 = X1.to(device)
            y1 = y1.to(device)
            fm = fm_label.to(device)
            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements
    
            _, fm_p1, _, rul_p1 = model(X0, X1)
    
            result[start:end, 0] = data_id
            result[start:end, 1] = fm_p1
            result[start:end, 2] = fm
            result[start:end, 3] = rul_p1
            result[start:end, 4] = y1

    result = tensor_2_numpy(result)
    result = pd.DataFrame(result, columns=['unit_id',
                                           'pred_fm',
                                           'fm',
                                           'pred_rul',
                                           'true_rul'])
    return result


# Define a custom KFold iterator to split at the unit level
class kFold:
    def __init__(self, df, n_splits):
        self.n_splits = n_splits
        self.unit_ids = df['id'].unique()

    def split(self, df):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for train_unit_ids, test_unit_ids in kf.split(self.unit_ids):
            train_units = df[df['id'].isin(train_unit_ids)]
            test_units = df[df['id'].isin(test_unit_ids)]
            yield train_units, test_units



###### Four evaluation metrics ######
def rmse_score(pred, true):
    '''
    Root Mean Square Error
    RMSE = \sqrt {\sum _{i=1}^n(pred_i- true_i)^2 / n }
    '''
    if len(pred) != len(true):
        raise RuntimeError
    
    if len(pred) == 0:
        return None
    
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(true, np.ndarray):
        true = np.array(true)

    return round(np.sqrt(((pred - true) ** 2).mean()),4)


def mae_score(pred, true):
    '''
    Mean absolute error
    '''
    if len(pred) != len(true):
        raise RuntimeError
    
    if len(pred) == 0:
        return None
    
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(true, np.ndarray):
        true = np.array(true)

    return abs(pred-true).mean()



def mape_score(pred, true):
    '''
    Mean Absolute Percentage Error 
    MAPE = {\sum_{i=1}^{n} |pred_t-true_t|/true_t} / n
    '''
    if len(pred) != len(true):
        raise RuntimeError
    
    if len(pred) == 0:
        return None
    
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(true, np.ndarray):
        true = np.array(true)
    
    epison = 1e-12 # to avoid the cases that epison is zero
    return round(np.abs((pred - true)/(true + epison)).mean(),4)


def health_score(pred, true):
    '''
    d[i] = pred[i] - true[i]
    if pred[i]<true[i], hs[i] = e^(d[i]/13)-1, else, hs[i] = e^(d[i]/10)-1
    hs = mean(hs[i]) for all i=1,2,...N
    '''
    if len(pred) != len(true):
        raise RuntimeError
    
    if len(pred) == 0:
        return None

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(true, np.ndarray):
        true = np.array(true)

    d = pred - true
    hs = map(lambda x: np.exp(-x / 13) - 1 if x < 0 else np.exp(x / 10) - 1, d)
    return round(np.array(list(hs)).mean(),4)



def monotonic_ratio(df, col = 'pred_rul'):
    '''
    Monotonic ratio of a df in a particular col
    '''
    count_total = 0
    count = 0
    for idx in range(1,101):
        df_unit = df.query(f'unit_id == {idx}').reset_index(drop=True)
        for i in range(1,len(df_unit)):
            count_total += 1
            if df_unit.loc[i,col]<=df_unit.loc[i-1,col]:
                count += 1
    return count/count_total
            


    





