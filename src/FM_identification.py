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
                                 max_iter=100, n_init=5, random_state=42) 
        
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


def trajectory_cluster_tube(train_2d_with_FMlabel, save_dir):
    '''
    Central path of two failure modes surrounded by tubes (by 1 std)
    '''
    df_fm_train_fm0 = train_2d_with_FMlabel.query('fm_label==0').reset_index(drop=True)
    df_fm_train_fm1 = train_2d_with_FMlabel.query('fm_label==1').reset_index(drop=True)
    
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
    plt.show()
    
    


if __name__ == "__main__":
    
    ds_name = 'FD003'        
    save_dir = f'../result-umap/{ds_name}-unsupervised-linearRUL-neighbors-80-minDist-1-2D'
    train_df = pd.read_csv(os.path.join(save_dir, f'{ds_name}-train.csv'))
    train_df_2d = pd.read_csv(os.path.join(save_dir, f'{ds_name}-2d-train.csv'))
    
    # get failure mode from train set
    _, train_2d_with_FMlabel = time_series_clustering(train_df, train_df_2d, 
                                         ds_name = ds_name,
                                         n_clusters = 2, 
                                         save_dir = save_dir)
    
    # plot_trajectory_by_fm(train_2d_with_FMlabel, save_dir)
    trajectory_cluster_tube(train_2d_with_FMlabel, save_dir)
        





