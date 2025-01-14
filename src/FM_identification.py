# -*- coding: utf-8 -*-
"""
Created on Sun May 28 16:11:45 2023

@author: Ying Fu
"""

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
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import os
import pandas as pd
import numpy as np
np.random.seed(42)

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

import plotly.graph_objects as go




def time_series_clustering(train, train_2d, ds_name = 'FD003',
                           n_clusters = 2, 
                           save_dir = '../result/figures'):
    '''
    For trajectories of training units, perfrom k means time series clustering
    to obtain the failure mode label of training units.
    Parameters:
        train: DataFrame, high-dimensional data. 
               Columns are: ['id','cycle','sensor1','sensor2',...'RUL']
        train_2d: DataFrame, low-dimensional data after UMAP dimension reduction
               Columns are: ['y1', 'y2','id', 'cycle', 'RUL', 'WC']
        ds_name: str, name of dataset
        n_clusters: User-specified parameter, the number of clusters
        plot: default=True, plot all trajectories and color by its label
    
    Return: DataFrame,
            Columns are: ['y1', 'y2','id', 'cycle', 'RUL', 'WC', 'fm_label']
    '''
        
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        train_2d_with_FMlabel = pd.read_csv(os.path.join(save_dir,  ds_name + '-2d-train_with_FMlabel.csv'))
        train_with_FMlabel = pd.read_csv(os.path.join(save_dir,  ds_name + '-train_with_FMlabel.csv'))
        print("-------load from existing-------")
    except FileNotFoundError:                    
        trajectories = []
        for idx, t in train_2d.groupby('id'):
            d = t[['y1','y2']]
            trajectories.append(d.values)
        
        trajectories = to_time_series_dataset(trajectories)
        scaler = TimeSeriesScalerMeanVariance()
        scaled_time_series = scaler.fit_transform(trajectories)
        
        # max_iter: Maximum number of iterations of the k-means algorithm for a single run.
        # n_init: Number of time the k-means algorithm will be run with different centroid seeds. 
        #         The final results will be the best output of n_init consecutive runs in terms of inertia.
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", 
                                 max_iter=1000, n_init=5, random_state=42) 
        
        cluster_labels = model.fit_predict(scaled_time_series)
        
        labels = pd.DataFrame()
        labels['id'] = pd.Series(train_2d['id'].unique())
        labels['fm_label'] = pd.Series(cluster_labels)
        
        train_2d_with_FMlabel = train_2d.merge(labels, on='id', how='outer')
        train_2d_with_FMlabel.to_csv(os.path.join(save_dir,  ds_name + '-2d-train_with_FMlabel.csv'), 
                                     index = False)

        train_with_FMlabel = train.merge(labels, on='id', how='outer')    
        train_with_FMlabel.to_csv(os.path.join(save_dir,  ds_name + '-train_with_FMlabel.csv'), 
                                     index = False)

    return train_with_FMlabel, train_2d_with_FMlabel
    

def plot_trajectory_by_fm(train_2d_with_FMlabel, save_dir):
    '''
    train_2d_with_FMlabel: DataFrame, 
        Columns are: ['y1', 'y2','id', 'cycle', 'RUL', 'WC', 'fm_label']

    Plot trajectories by failure modes after trajectory clustering
        Each line is a unit and the color is its corresponding failure mode
    '''    
    # specify the color
    color_map = {0: 'blue', 1: 'red', 2: 'green'}
    train_2d_with_FMlabel['color'] = train_2d_with_FMlabel['fm_label'].map(color_map)
    
    # First figure: each line is a unit and the color is its corresponding failure mode
    id_lst = train_2d_with_FMlabel['id'].unique().tolist()
    fig = go.Figure()
    for idx in id_lst:
        df_unit = train_2d_with_FMlabel.query(f'id=={idx}')
        fig.add_trace(go.Scatter3d( x = df_unit['RUL'] , 
                                    y = df_unit['y1'], 
                                    z = df_unit['y2'],
                                    mode ='lines',
                                    line = dict(color = df_unit['color'], width=5),
                                    name = str(idx)))    
    fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title=' '* 30 +'RUL',
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern')
                ),
                yaxis=dict(
                    title=' '* 30 +'y1',
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern')  
                ),
                zaxis=dict(
                    title=' '* 30 +'y2',
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern')
                )
            )
        ) 
    fig.write_html(os.path.join(save_dir,'trajectoryByUnit-color-fm.html'))


def __agg_mean_std(df):
    '''
    aggregate mean and std
    '''
    agg_df = df.groupby(['RUL'], as_index=False).agg(
                        {'y1':['mean','std'],'y2':['mean', 'std']}).reset_index(drop=True)
    agg_df.columns = agg_df.columns.map('_'.join).str.strip('_')
    agg_df = agg_df.fillna(0)
    
    return agg_df


def __tube_surface(ax, points, color='r', alpha=0.5, 
                 n_angles = 90, ind_x = 2, ind_y = 0, ind_z = 1, ind_r = 3):
    e = 0.000001
    points = points[0::3,:]
    for i in range(len(points)-1):
        x = points[i:(i+2),ind_x]
        cy = points[i:(i+2),ind_y]
        cz = points[i:(i+2),ind_z]
        r = points[i:(i+2),ind_r]
        if r[0] == 0 and r[1] == 0:
              r[1] = e
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)[..., np.newaxis]  # (n_angles, 1)

        # divide the side surface of trapzoidal cylinder into n_angles trapzoids
        Y = r * np.cos(angles) + cy   # Y[i,j] = cos(angles[i,1]) * r[j] + cy[j]
        Z = r * np.sin(angles) + cz
        X = np.zeros(Y.shape) + x

        # divide the side surface of trapzoidal cylinder into n_angles trapzoids, then divid a trapzoid into two triangles
        # circle0 circle1
        #   0     0'     triangle: 0 1  0'   i,j, i'  j = (i+1) % len
        #   1     1'     triangle: 1 1' 0'   j,j',i'
        #   2     2'
        #   ...
        #   n     n'
        tri = []
        for i in range(n_angles):
            n = (i+1) % n_angles
            if Y[i,0] != Y[n,0] or Z[i,0] != Z[n,0] or X[i,0] != X[n,0]:
                tri.append([i*2,n*2,i*2+1])
            if Y[i,1] != Y[n,1] or Z[i,1] != Z[n,1] or X[i,1] != X[n,1]:
                tri.append([n*2,n*2+1,i*2+1])
        tri = np.array(tri)
        
        ax.plot_trisurf(X.flatten(), Y.flatten(), Z.flatten(), triangles = tri, linewidth=0.2, antialiased=True, color=color, alpha=alpha)


def trajectory_cluster_tube(ds_name, train_2d_with_FMlabel, save_dir):
    '''
    Central path of two failure modes surrounded by tubes (by 1 std)
    '''    
    
    df_fm_train_fm0 = train_2d_with_FMlabel.query('fm_label==0').reset_index(drop=True)
    df_fm_train_fm1 = train_2d_with_FMlabel.query('fm_label==1').reset_index(drop=True)
    
    fm0_units = df_fm_train_fm0['id'].unique()
    fm1_units = df_fm_train_fm1['id'].unique()
    print(f'{len(fm0_units) = }')
    print(f'{len(fm1_units) = }')
    print(f'{fm0_units = }, {fm1_units = }')
    
    agg_dr_fm0 = __agg_mean_std(df_fm_train_fm0)
    agg_dr_fm1 = __agg_mean_std(df_fm_train_fm1)

    agg_dr_fm0['r_s'] = np.sqrt((1*agg_dr_fm0['y1_std'])**2 + (1*agg_dr_fm0['y2_std'])**2)    
    agg_dr_fm1['r_s'] = np.sqrt((1*agg_dr_fm1['y1_std'])**2 + (1*agg_dr_fm1['y2_std'])**2)
    
    # avg +/- 1 std
    path0 = agg_dr_fm0[['y1_mean', 'y2_mean', 'RUL','r_s']].values
    path1 = agg_dr_fm1[['y1_mean', 'y2_mean', 'RUL','r_s']].values

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot(agg_dr_fm0['RUL'], agg_dr_fm0['y1_mean'], agg_dr_fm0['y2_mean'],color='b')
    ax.plot(agg_dr_fm1['RUL'], agg_dr_fm1['y1_mean'], agg_dr_fm1['y2_mean'],color='r')
    ax.set_xlabel(' '* 30 +'RUL',fontsize=32,labelpad=18)
    ax.set_ylabel(' '* 30 +'y1',fontsize=32,labelpad=18)
    ax.set_zlabel(' '* 30 +'y2',fontsize=32,labelpad=18)
    
    __tube_surface(ax, path0, alpha=0.2, color='b')
    __tube_surface(ax, path1, alpha=0.2, color='r')
    plt.savefig(os.path.join(save_dir, f'trajectory-tube-{ds_name}.pdf'))
    plt.savefig(os.path.join(save_dir, f'trajectory-tube-{ds_name}.svg'))
    plt.show()
    
    

    
def find_equal_value_keys(data_dict):
    '''
    Identify the keys that have the same value in a dictionary. 
    Convert each value to a tuple to make it hashable, then use this tuple 
    as the key in another dictionary.        
    '''
    value_to_keys = {}    
    for key, value in data_dict.items():
        value_tuple = tuple(value) if isinstance(value, list) else tuple(value.tolist()) 
        if value_tuple in value_to_keys:
            value_to_keys[value_tuple].append(key)
        else:
            value_to_keys[value_tuple] = [key]
    
    equal_value_keys_with_values = [(keys, list(value_tuple)) for value_tuple, keys in value_to_keys.items() if len(keys) > 1]
    
    return equal_value_keys_with_values


def FD004_FM_train():
    '''
    Determine the failure mode for the training units in dataset FD004.
    '''
    ds_name = 'FD004'
    fm0_dict = {}
    fm1_dict = {}
    for WC in range(6):
        print(f'{   WC = }')
        save_dir = f'../result-umap-0730-2024/WC_{WC}-{ds_name}-unsupervised-linearRUL-neighbors-80-minDist-1-2D'
        train_df = pd.read_csv(os.path.join(save_dir, f'{ds_name}-train.csv'))
        train_df_2d = pd.read_csv(os.path.join(save_dir, f'{ds_name}-2d-train.csv'))
        
        # get failure mode from train set
        _, train_2d_with_FMlabel = time_series_clustering(train_df, train_df_2d, 
                                             ds_name = ds_name,
                                             n_clusters = 2, 
                                             save_dir = save_dir)
        df_fm_train_fm0 = train_2d_with_FMlabel.query('fm_label==0').reset_index(drop=True)
        df_fm_train_fm1 = train_2d_with_FMlabel.query('fm_label==1').reset_index(drop=True)
        
        fm0_units = df_fm_train_fm0['id'].unique()
        fm1_units = df_fm_train_fm1['id'].unique()
        
        fm0_dict[WC] = fm0_units
        fm1_dict[WC] = fm1_units
        
        plot_trajectory_by_fm(train_2d_with_FMlabel, save_dir)        

    equal_keys0 = find_equal_value_keys(fm0_dict)
    print(equal_keys0)

    equal_keys1 = find_equal_value_keys(fm1_dict)
    print(equal_keys1)



def FD003_FM_train():
    '''
    Determine the failure mode for the training units in dataset FD004.
    '''
    ds_name = 'FD003'        
    save_dir = f'../result-umap/{ds_name}-unsupervised-linearRUL-neighbors-80-minDist-1-2D'
    train_df = pd.read_csv(os.path.join(save_dir, f'{ds_name}-train.csv'))
    train_df_2d = pd.read_csv(os.path.join(save_dir, f'{ds_name}-2d-train.csv'))
    
    # get failure mode from train set
    _, train_2d_with_FMlabel = time_series_clustering(train_df, train_df_2d, 
                                          ds_name = ds_name,
                                          n_clusters = 2, 
                                          save_dir = save_dir)
    
    plot_trajectory_by_fm(train_2d_with_FMlabel, save_dir)
    trajectory_cluster_tube('FD003', train_2d_with_FMlabel, save_dir)

    

def robustness_umap_fm(ds_name = 'FD003'): 
    
    '''
    To verify the robustness of dimension reduction technique, UMAP, 
    we consider additional cases where we randomly retain 70%, 50%, 30%, 20%, and 10% 
    of the total training units. We then perform UMAP dimension reduction on these 
    reduced datasets and utilize the proposed time series clustering to identify 
    the failure modes within these units. 
    Finally, we compare the failure modes identified in these reduced datasets with 
    those identified from the full dataset, which includes all training units. 
    We record the number of units misclassified between these two scenarios, 
    such as those labeled as failure mode 1 in the reduced datasets but actually 
    categorized as failure mode 0 in the full dataset, and vice versa.
    '''
        
    # failure mode labels for training units
    fm0_unit_id_lst = [1.,   3.,   4.,   5.,   6.,   8.,  12.,  13.,  14.,  15.,  22.,
                 23.,  25.,  26.,  28.,  29.,  30.,  31.,  32.,  35.,  36.,  40.,
                 44.,  47.,  48.,  50.,  51.,  52.,  53.,  54.,  56.,  58.,  61.,
                 63.,  64.,  65.,  66.,  67.,  68.,  69.,  70.,  74.,  76.,  78.,
                 79.,  80.,  83.,  86.,  87.,  90.,  91.,  92.,  93.,  95.,  99., 
                 100.]
    
    fm1_unit_id_lst = [ 2.,  7.,  9., 10., 11., 16., 17., 18., 19., 20., 21., 24., 27.,
                 33., 34., 37., 38., 39., 41., 42., 43., 45., 46., 49., 55., 57.,
                 59., 60., 62., 71., 72., 73., 75., 77., 81., 82., 84., 85., 88.,
                 89., 94., 96., 97., 98.]
    
    n_components = 2
    n_neighbors = 80
    min_dist = 1
    result_dict = {}
    for ratio in [0.3, 0.5, 0.7, 0.8, 0.85, 0.9]:
        save_dir = f'../result-umap-0722-2024/{ds_name}-dropRatio-{ratio}-unsupervised-linearRUL-neighbors-{n_neighbors}-minDist-{min_dist}-{n_components}D'           
        train_df = pd.read_csv(os.path.join(save_dir, f'{ds_name}-train.csv'))
        train_df_2d = pd.read_csv(os.path.join(save_dir, f'{ds_name}-2d-train.csv'))
        
        _, train_2d_with_FMlabel = time_series_clustering(train_df, train_df_2d, 
                                              ds_name = ds_name,
                                              n_clusters = 2, 
                                              save_dir = save_dir)
        df_fm_train_fm0 = train_2d_with_FMlabel.query('fm_label==0').reset_index(drop=True)
        df_fm_train_fm1 = train_2d_with_FMlabel.query('fm_label==1').reset_index(drop=True)
        
        fm0_units = df_fm_train_fm0['id'].unique()
        fm1_units = df_fm_train_fm1['id'].unique()
    
        # Count the matches
        count1_1 = sum(1 for i in fm0_units if i in fm0_unit_id_lst) 
        count1_2 = sum(1 for j in fm1_units if j in fm1_unit_id_lst)
        count2_1 = sum(1 for i in fm0_units if i in fm1_unit_id_lst) 
        count2_2 = sum(1 for j in fm1_units if j in fm0_unit_id_lst)
    
        
        count = max(count1_1 + count1_2, count2_1 + count2_2)
        print(f'count = {count}')
        result_dict[ratio] = count
    
    return result_dict
    
    


if __name__ == "__main__":
    
    FD003_FM_train()
    FD004_FM_train()
    robustness_umap_fm()
    


    























