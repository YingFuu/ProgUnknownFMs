# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:35:28 2023

@author: Ying Fu
"""

import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("ticks")
import os
from train_evaluate import rmse_score, mape_score, health_score, monotonic_ratio, mae_score
from matplotlib import pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
BIGGER_SIZE = 36
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=24)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



def clip(df, col_name_lst, threshold=125):
    '''
    Clip the values in the columns of the dataframe (df) in col_name_lst without exceeding threshold.
    '''
    if threshold == None:
        return df
    else:
        for col_name in col_name_lst:
            df[col_name] = df[col_name].apply(lambda x: min(x,threshold))
        return df



def reg_result_overall(save_dir):
    '''
    Given a result save dir, get the overall regression result.
    '''

    def get_result(r):
        rmse_lst = []
        mape_lst=[]
        mae_lst=[]
        hs_lst=[]
        mono_lst=[]     # only used for the whole sequence

        rmse_last_lst = []
        mape_last_lst =[]
        mae_last_lst = []
        hs_last_lst = []
        for j, fold in enumerate([0,1,2,3,4]):
            df = pd.read_csv(os.path.join(save_dir, f'fold-{fold}-result-{r}.csv'))
            df_copy = df.copy()
            df_clip = clip(df_copy, col_name_lst=['pred_rul','true_rul'], threshold=125)
            df_clip_lst = df_clip.drop_duplicates('unit_id', keep='last').reset_index(drop=True)

            rmse = rmse_score(df['pred_rul'], df['true_rul'])
            mape = mape_score(df['pred_rul'], df['true_rul'])
            mae = mae_score(df['pred_rul'], df['true_rul'])
            hs = health_score(df['pred_rul'], df['true_rul'])
            mono_ratio = monotonic_ratio(df)
            rmse_lst.append(rmse)
            mape_lst.append(mape)
            mae_lst.append(mae)
            hs_lst.append(hs)
            mono_lst.append(mono_ratio)

            rmse_last = rmse_score(df_clip_lst['pred_rul'], df_clip_lst['true_rul'])
            mape_last = mape_score(df_clip_lst['pred_rul'], df_clip_lst['true_rul'])
            mae_last = mae_score(df_clip_lst['pred_rul'], df_clip_lst['true_rul'])
            hs_last = health_score(df_clip_lst['pred_rul'], df_clip_lst['true_rul'])
            rmse_last_lst.append(rmse_last)
            mae_last_lst.append(mae_last)
            mape_last_lst.append(mape_last)
            hs_last_lst.append(hs_last)

        return round(np.mean(rmse_lst),4),          round(np.std(rmse_lst),4),\
               round(np.mean(mape_lst),4),          round(np.std(mape_lst),4), \
               round(np.mean(mae_lst), 4),          round(np.std(mae_lst), 4), \
               round(np.mean(hs_lst),4),            round(np.std(hs_lst),4), \
               round(np.mean(rmse_last_lst),4),     round(np.std(rmse_last_lst),4),\
               round(np.mean(hs_last_lst), 4),      round(np.std(hs_last_lst), 4), \
               round(np.mean(mape_last_lst), 4),    round(np.std(mape_last_lst),4), \
               round(np.mean(mae_last_lst), 4),     round(np.std(mae_last_lst), 4), \
               round(np.mean(mono_lst), 4),         round(np.std(mono_lst),4)

    r_lst = ['train', 'var', 'test']
    result  = {}
    for r in r_lst:
        result[r] = get_result(r)
    result_df = pd.DataFrame(result)
    result_df.index = [  'rmse_mean',           'rmse_std',
                         'mape_mean',           'mape_std',
                         'mae_mean',            'mae_std',
                         'hs_mean',             'hs_std',
                         'rmse_last_mean',      'rmse_last_std',
                         'hs_last_mean',        'hs_last_std',
                         'mape_last_mean',      'mape_last_std',
                         'mae_last_mean',       'mae_last_std',
                         'mono_mean',           'mono_std']
    result_df.columns = r_lst
    return result_df



def result_summarize(save_dir):
    def __add_time_step(df):
        '''
        for each unit, add the time step for each unit.
        '''
        df_time_step_lst = []
        for unit_id in range(1,101):
            df_unit = df.query(f'unit_id == {unit_id}')
            df_unit_copy = df_unit.copy()
            df_unit_copy['time_step'] = pd.Series([i for i in range(len(df_unit))], dtype=float)
            df_time_step_lst.append(df_unit_copy)
        df_time_step_lst = pd.concat(df_time_step_lst).reset_index(drop=True)
        return df_time_step_lst

    df_lst = []
    for fold in [0,1,2,3,4]:
        df = pd.read_csv(os.path.join(save_dir, f'fold-{fold}-result-test.csv'))
        df1 = __add_time_step(df)
        df_copy = df1.copy()
        # calculate the error for each sample
        df_copy['error']= df_copy['pred_rul'] - df_copy['true_rul']
        df_copy['fold'] = fold
        df_copy['idx'] = pd.Series([i for i in range(len(df_copy))])
        df_copy['abs_error'] = np.abs(df_copy['error'])
        df_copy['rmse_error'] = np.sqrt(df_copy['error']**2)
        df_copy['hs'] = df_copy['error'].apply(lambda x: np.exp(-x / 13) - 1 if x < 0 else np.exp(x / 10) - 1)
        df_copy['mape'] = np.abs(df_copy['error'])/df_copy['true_rul']
        df_lst.append(df_copy)
    df_lst = pd.concat(df_lst).reset_index(drop=True) # concate all five fold result
    return df_lst



def result_plot_new(save_dir_lst,save_path = None):
    '''
    For each unit, plot the prediction result on the same figure.
    df_lst = [(df_1, label_1, linestyle_1),..., (df_n, label_n, linestyle_n)]
    '''
    os.makedirs(save_path, exist_ok=True)
    dataframes = []
    for (src_name, save_dir) in save_dir_lst:
        for fold in [0, 1, 2, 3, 4]:
            df = pd.read_csv(os.path.join(save_dir, f'fold-{fold}-result-test.csv'))
            for uint_id in range(1, 101):
                df_unit = df.query(f'unit_id == {uint_id}').reset_index(drop=True)
                df_unit_copy = df_unit.copy()
                if len(df_unit) == 0:
                    continue
                cycle_length_lst = [i for i in range(len(df_unit))]
                df_unit_copy['cycle'] = pd.Series(cycle_length_lst)
                df_unit_copy['fold'] = fold
                df_unit_copy['model'] = src_name
                dataframes.append(df_unit_copy)
    dataframes = pd.concat(dataframes).reset_index(drop=True)

    for unit_id in range(1,101):
        df_unit = dataframes.query(f'unit_id == {unit_id}').reset_index(drop=True)
        if len(df_unit) == 0:
            continue
        plt.figure()
        for src_name in df_unit['model'].unique():
            sub = df_unit[df_unit['model']==src_name].reset_index(drop=True)
            plt.plot(sub['cycle'] + 60, sub['true_rul'], linewidth=3, c='black')
            sub_stat = sub.groupby('cycle')['pred_rul'].agg(['mean', 'std']).reset_index()
            plt.plot(sub_stat['cycle']+60, sub_stat['mean'], linewidth=3, label=f'{src_name}')
            plt.fill_between(sub_stat['cycle']+60, sub_stat['mean'] - sub_stat['std'], sub_stat['mean'] + sub_stat['std'],
                             alpha=0.2)
        plt.xlabel('cycle time')
        plt.ylabel('RUL')
        plt.title(f'test unit: {unit_id}')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'{unit_id}.pdf'), bbox_inches='tight')
        plt.show()




def box_plot_analysis(save_dir_lst, start, end, step):
    '''
    plot the box plot of models.
    '''
    def __get_data(save_dir, src_name):
        df_lst = result_summarize(save_dir)
        df_mean = df_lst.groupby(
            ['unit_id','idx']
            ).agg({
            'pred_rul': ['mean', 'std'],
            'true_rul': ['mean', 'std'],
            'error': ['mean', 'std'],
            'abs_error':['mean', 'std'],
            'rmse_error':['mean', 'std'],
            'hs': ['mean', 'std'],
            'mape': ['mean', 'std']
            }).reset_index()
        df_mean.columns = df_mean.columns.map('_'.join).str.strip('_')


        df_mean_interval = []
        for i, left in enumerate(range(start, end, step)):
            right=left + step
            df_sub = df_mean.query(f'true_rul_mean>{left} & true_rul_mean<={right}' ).reset_index(drop=True)
            df_sub['left'] = left
            df_sub['right'] = right
            df_sub['range'] = f'${left} <RUL \leq {right}$'
            df_mean_interval.append(df_sub)
        df_mean_interval = pd.concat(df_mean_interval).reset_index(drop=True)
        df_mean_interval['model'] = src_name
        return df_mean_interval


    df_interval_lst = []
    for save_dir, src_name in save_dir_lst:
        df_interval = __get_data(save_dir, src_name)
        df_interval_lst.append(df_interval)
    df_interval = pd.concat(df_interval_lst).reset_index(drop=True)
    plt.figure(figsize=(21, 12))  # Adjust the figure size if needed
    sns.boxplot(x='range', y='abs_error_mean', hue='model', data=df_interval)
    plt.xlabel('Actual RUL')
    plt.xticks(rotation = 25)
    plt.ylabel('Absolute Error')
    plt.show()

  

def plot_prob_estimation(save_dir):
    '''
    For each unit, plot the FM probability estimation
    '''
    dataframes = []
    for fold in [0,1,2,3,4]:
        df = pd.read_csv(os.path.join(save_dir, f'fold-{fold}-result-test.csv'))
        for uint_id in range(1,101):
            df_unit = df.query(f'unit_id == {uint_id}').reset_index(drop=True)
            df_unit_copy = df_unit.copy()
            if len(df_unit)==0:
                continue
            cycle_length_lst = [i for i in range(len(df_unit))]
            df_unit_copy['cycle'] = pd.Series(cycle_length_lst)
            df_unit_copy['fold'] = fold
            dataframes.append(df_unit_copy)
    dataframes = pd.concat(dataframes).reset_index(drop=True)


    for unit_id in range(1,101):
        df_unit = dataframes.query(f'unit_id == {unit_id}').reset_index(drop=True)
        if len(df_unit) == 0:
            continue
        unit_statistics = df_unit.groupby('cycle')['pred_fm'].agg(['mean', 'std']).reset_index()

        if unit_id == 87:
            plt.plot(unit_statistics['cycle']+60, unit_statistics['mean'], linewidth=5, 
                     label=f'test unit: {unit_id}', c='g', linestyle = '--')
        elif unit_id == 97:
            plt.plot(unit_statistics['cycle']+60, unit_statistics['mean'], linewidth=5, 
                     label=f'test unit: {unit_id}', c='r', linestyle = 'dashdot')
        elif unit_id == 1:
            plt.plot(unit_statistics['cycle']+60, unit_statistics['mean'], linewidth=5, 
                     c = 'black', linestyle = ':', label = 'other test units')
        else:
            plt.plot(unit_statistics['cycle']+60, unit_statistics['mean'], linewidth=5, 
                     c = 'black', linestyle = ':')

        # Shade regions representing standard deviation
        plt.fill_between(unit_statistics['cycle'], unit_statistics['mean']-unit_statistics['std'],
                         unit_statistics['mean']+unit_statistics['std'],
                         alpha=0.25)
        plt.xlabel('cycle time')
        plt.ylabel('Real-time FM probability estimation')
        plt.legend()
        plt.xlim(30, 500)
        plt.ylim(-0.1, 1.1)
        save_path = os.path.join(save_dir, 'prob_plot')
        os.makedirs(save_path, exist_ok=True)
    plt.savefig("../latex/figures/prob-estimation.pdf", 
                    format="pdf", bbox_inches="tight")
    plt.show()
    
    
    

  
if __name__ == "__main__":
    src_name1 = 'LSTM'
    save_dir1 = '../result-FD003-1210/hs/LSTM-h1-32-h2-32'

    src_name2 = r"Joint-LSTM ($\eta = 0$)"
    save_dir2 = '../result/result-FD003-1210/hs/LSTM-joint-h1-16-h2-16'
    
    src_name3 = r"Joint-LSTM ($\eta = 0.1$)"
    save_dir3 = '../result/result-FD003-1210/hs-lambda-0.1/LSTM-joint-grc-h1-16-h2-16'
    
    src_name4 = r"Joint-LSTM ($\eta = 0.5$)"
    save_dir4 = '../result/result-FD003-1210/hs-lambda-0.5/LSTM-joint-grc-h1-16-h2-16'
    
    src_name5 = r"Joint-LSTM ($\eta = 1$)"
    save_dir5 = '../result/result-FD003-1210/hs-lambda-1/LSTM-joint-grc-h1-16-h2-16'

    src_name6 = r"Joint-LSTM ($\eta = 2$)"
    save_dir6 = '../result/result-FD003-1210/hs-lambda-2/LSTM-joint-grc-h1-16-h2-16'
    
    src_name7 = r"Joint-LSTM ($\eta = 4$)"
    save_dir7 = '../result/result-FD003-1210/hs-lambda-4/LSTM-joint-grc-h1-16-h2-16'

    result_plot_new([(src_name2, save_dir2),
                     (src_name3, save_dir3),
                     (src_name4, save_dir4),
                     (src_name5, save_dir5),
                     (src_name6, save_dir6),
                     (src_name7, save_dir7)],
                     save_path='../result/Figure-23-12-12-ntw-60')
    
    box_plot_analysis([(save_dir1, src_name1),(save_dir2, src_name2),(save_dir4, src_name4)],
                        start = 0, end = 350, step = 50)
 
    plot_prob_estimation(save_dir2)











