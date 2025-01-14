# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:52:26 2023

@author: Ying Fu
"""

import numpy as np
np.random.seed(0)
import random
random.seed(0)

import pandas as pd
import os
import time
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
from umap import UMAP

from data.JetEngine import prepare_data


import plotly.express as px # interactive visualization
import plotly.graph_objects as go


def dr_umap(train_df, test_df, ds_name,
            save_path = '../result/figures',
            normalize = True,
            drop_columns = ['id','cycle',
                            'op1','op2','op3','WC'],
            supervised = False,
            n_neighbors = 80, n_components = 3, metric = 'euclidean', 
            min_dist = 0.1):
    '''
    Parameters
    ----------
    train_df : DataFrame.
    test_df : DataFrame.
    ds_name: str
    save_dir : str.
        Directory to save the result. The default is '../result/figures'.
    normalize : Boolean, optional
        Whether normalize the data while performing UMAP. The default is True.
    drop_columns : list, optional
        Columns that are not contained while performing UMAP. 
        The default is ['id','cycle','op1','op2','op3','time_id'].
    supervised : Boolean, optional
        Do supervised UMAP or not. The default is True.
    n_neighbors : int, optional
        the number of approximate nearest neighbors used to construct the initial 
        high-dimensional graph. It effectively controls how UMAP balances local 
        versus global structure - low values will push UMAP to focus more on 
        local structure by constraining the number of neighboring points 
        considered when analyzing the data in high dimensions, 
        while high values will push UMAP towards representing the 
        big-picture structure while losing fine detail. The default is 80.
    n_components : int, optional
        The dimension of the space to embed into. The default is 3.
    metric : TYPE, optional
        DESCRIPTION. The default is 'euclidean'.
    min_dist : float, optional
        minimum distance between points in low-dimensional space. 
        This parameter controls how tightly UMAP clumps points together, 
        with low values leading to more tightly packed embeddings. 
        Larger values of min_dist will make UMAP pack points together more 
        loosely, focusing instead on the preservation of the broad topological 
        structure. The default is 0.1.

    Raises
    ------
    RuntimeError
        We will try to first load the existing train/test dataset after UMAP.
        If the file doesn't exist, then perform UMAP.

    Returns
    -------
    umap_train_df : DataFrame
        Dataframe of train set after dimension reduction.
    umap_test_df : DataFrame
        Dataframe of test set after dimension reduction.

    '''
    
    # save_path = os.path.join(save_dir, f'{ds_name}')
    os.makedirs(save_path, exist_ok=True)
    
    # save the high dimensional train and test dataset
    train_df.to_csv(os.path.join(save_path, ds_name + '-train.csv'), index=False)
    test_df.to_csv(os.path.join(save_path, ds_name + '-test.csv'), index=False)
     
    try:
        # try to load the existing train and test datasets after UMAP
        umap_train_df = pd.read_csv(os.path.join(save_path, ds_name + '-' + str(n_components) + 'd-train.csv'))
        umap_test_df = pd.read_csv(os.path.join(save_path, ds_name + '-' + str(n_components) + 'd-test.csv'))
        print("-------load from existing-------")
    except FileNotFoundError:
        
        print("-------performing UMAP----------")
        train_df_copy = train_df.copy()
        test_df_copy = test_df.copy()
        
        for col in drop_columns:
            if col in list(train_df_copy.columns):
                train_df_copy.drop(col, axis=1, inplace=True)
            if col in list(test_df_copy.columns):
                test_df_copy.drop(col, axis=1, inplace=True)
        
        if normalize:
            columns_to_drop = []   # drop columns if necessary (In the case of performing UMAP on a subset of FD002 and FD004 under one working condition)
            for column in list(train_df_copy.columns):
                train_min = train_df[column].min()
                train_max = train_df[column].max()
                
                if train_max - train_min == 0:
                    columns_to_drop.append(column)
                    continue
                    
                train_df_copy[column] = (train_df_copy[column] - train_min) / (train_max - train_min)
                test_df_copy[column] = (test_df_copy[column] - train_min) / (train_max - train_min)
        
        if columns_to_drop:
            train_df_copy.drop(columns=columns_to_drop, inplace=True)
            test_df_copy.drop(columns=columns_to_drop, inplace=True)
            print(f'Dropped columns: {columns_to_drop}')
        
        train_df_X = train_df_copy.drop('RUL', axis=1, inplace=False)
        train_df_y = train_df_copy['RUL']
        
        test_df_X = test_df_copy.drop('RUL', axis=1, inplace=False)
        
        if set(list(train_df_X.columns)) != set(list(test_df_X.columns)):
            raise RuntimeError
        
        # Configure UMAP hyperparameters
        reducer = UMAP(n_neighbors=n_neighbors,
                        # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
                        n_components=n_components,  # default 2, The dimension of the space to embed into.
                        metric=metric,
                        # default 'euclidean', The metric to use to compute distances in high dimensional space.
                        n_epochs=1000,
                        # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
                        learning_rate=1.0,  # default 1.0, The initial learning rate for the embedding optimization.
                        init='spectral',
                        # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
                        min_dist=min_dist,  # default 0.1, The effective minimum distance between embedded points.
                        spread=1,
                        # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
                        low_memory=False,
                        # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
                        set_op_mix_ratio=1.0,
                        # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
                        local_connectivity=1,
                        # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
                        repulsion_strength=1.0,
                        # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
                        negative_sample_rate=5,
                        # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
                        transform_queue_size=4.0,
                        # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
                        a=None,
                        # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                        b=None,
                        # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                        random_state=42,
                        # default: None, If int, random_state is the seed used by the random number generator;
                        metric_kwds=None,
                        # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
                        angular_rp_forest=False,
                        # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
                        target_n_neighbors=-1,
                        # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
                        # target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different.
                        # target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
                        # target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
                        transform_seed=42,
                        # default 42, Random seed used for the stochastic aspects of the transform operation.
                        verbose=False,  # default False, Controls verbosity of logging.
                        unique=False,
                        # default False, Controls if the rows of your data should be uniqued before being embedded.
                        )
        
        
        
        if supervised:
            # Fit and transform the train data 
            X_train_trans = reducer.fit_transform(train_df_X,train_df_y)
        else:
            X_train_trans = reducer.fit_transform(train_df_X)
        
        # Fit and transform the test data 
        X_test_trans = reducer.transform(test_df_X) 
    
        if X_train_trans.shape[0] != train_df_X.shape[0]:
            raise RuntimeError
        if X_train_trans.shape[1] != n_components:
            raise RuntimeError
        if X_test_trans.shape[0] != test_df_X.shape[0]:
            raise RuntimeError
        if X_test_trans.shape[1] != n_components:
            raise RuntimeError
    
        umap_train_df = pd.DataFrame(X_train_trans, columns=['y' + str(i+1) for i in range(n_components)])
        umap_train_df['id'] = train_df['id']
        umap_train_df['cycle'] = train_df['cycle']
        umap_train_df['RUL'] = train_df['RUL']
        umap_train_df['WC'] = train_df['WC']
        umap_train_df.to_csv(os.path.join(save_path, ds_name + '-' + str(n_components) + 'd-train.csv'), index = False)

        umap_test_df = pd.DataFrame(X_test_trans, columns=['y' + str(i+1) for i in range(n_components)])
        umap_test_df['id'] = test_df['id']
        umap_test_df['cycle'] = test_df['cycle']
        umap_test_df['RUL'] = test_df['RUL']
        umap_test_df['WC'] = test_df['WC']
        umap_test_df.to_csv(os.path.join(save_path, ds_name + '-' + str(n_components) + 'd-test.csv'), index = False)
        
    return umap_train_df, umap_test_df


def umap_plot_2d_and_add_time(umap_train_df_2d, umap_test_df_2d, 
                              save_dir = '../result/figures',
                              ds_name = 'FD003'):
    '''
    Perform UMAP to reduce the dimension to 2D and then add an additional time 
    axis to each unit, then visualize each unit in 3D line plot. 
    
    Parameters
    ----------
    umap_train_df_2d : DataFrame
        Train set in 2D low dimensional space.
    umap_test_df_2d : DataFrame
        Test set in 2D low dimensional space.
    save_dir : str, optional
        Directory to save the results. The default is '../result/figures'.
    ds_name : str, optional
        The default is 'FD003'.

    Raises
    ------
    RuntimeError
        Chech the dimensionality of train and test datasets in both high and low dimensional space.

    Returns
    -------
    None.
    '''
    os.makedirs(save_dir, exist_ok=True) 
    
    # The scatter plot of train and test dataset in 2D
    # train
    cm = plt.cm.get_cmap('RdYlBu')
    plt.scatter(umap_train_df_2d['y1'], umap_train_df_2d['y2'], c = umap_train_df_2d['RUL'], cmap=cm)
    clb=plt.colorbar()
    clb.ax.tick_params(labelsize=BIGGER_SIZE) 
    clb.ax.set_title('RUL',fontsize=BIGGER_SIZE)
    plt.xlabel('y1', fontsize=BIGGER_SIZE, fontname='Times New Roman')  # Set the font size and family for the x-axis label
    plt.ylabel('y2', fontsize=BIGGER_SIZE, fontname='Times New Roman')  # Set the font size and family for the x-axis label
    plt.savefig(os.path.join(save_dir, f'train-2D-Scatter-{ds_name}.pdf'))
    plt.show()

    # test
    plt.scatter(umap_test_df_2d['y1'], umap_test_df_2d['y2'])
    plt.savefig(os.path.join(save_dir, f'test-2D-Scatter-{ds_name}.pdf'))
    plt.show()
    
    # 3D line plot: plot within each working condition
    wc_lst = list(umap_train_df_2d['WC'].unique())
    print(f"{wc_lst = }")
    for wc in wc_lst:
        wc_new_df_train = umap_train_df_2d.query(f'WC=={wc}').reset_index(drop=True)
        
        # plot the train 
        fig_umap_train = px.line_3d(wc_new_df_train, x='RUL', y='y1', z='y2',color='id',
                              height=900, width=900).update_traces(marker=dict(size=3, opacity = 0.5))
        fig_train = go.Figure(data=fig_umap_train.data, layout = fig_umap_train.layout)
        fig_train.update_layout(
            scene=dict(
                xaxis=dict(
                    title=' '* 30 +'RUL',   # adjust the space between axis labels and plot area
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern'), 
                ),
                yaxis=dict(
                    title=' '* 30 + 'y1',
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern'),  
                ),
                zaxis=dict(
                    title=' '* 30 +'y2',
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern'), 
                ) 
            ),
        )
        fig_train.write_html(os.path.join(save_dir,'trajectory_train_all_'+ ds_name + '-wc' + str(wc) + '-UMAP-3Dline.html'))




def umap_plot_3d_RUL(umap_train_df_3d, umap_test_df_3d, 
                    save_dir = '../result/23-07-21',
                    ds_name = 'FD003'):
    '''
    Ignore the time axis and perform UMAP to reduce the dimension to 3D and then visualize the RUL. 

    Parameters
    ----------
    umap_train_df_3d : DataFrame
        Train set in 3D low dimensional space.
    umap_test_df_3d : DataFrame
        Test set in 3D low dimensional space.
    save_dir : str, optional
        Directory to save the results. The default is '../result/23-07-21'.
    ds_name : str, optional
        The default is 'FD003'.

    Returns
    -------
    None.
    '''
    os.makedirs(save_dir, exist_ok=True)
    fig_train = px.scatter_3d(umap_train_df_3d, x='y1', y='y2', z='y3', 
                         color=umap_train_df_3d['RUL'],color_continuous_scale='RdYlBu',
                         height=900, width=900).update_traces(marker=dict(size=2))
    fig1 = go.Figure(data=fig_train.data, layout = fig_train.layout)
    fig1.update_layout(
            scene=dict(
                xaxis=dict(
                    title=' '* 30 +'y1',
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern')
                ),
                yaxis=dict(
                    title=' '* 30 +'y2',
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern')  
                ),
                zaxis=dict(
                    title=' '* 30 +'y3',
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern')
                )
            )
        )

    fig1.write_html(os.path.join(save_dir, 'RUL.html'))


def umap_plot_2d_WC(umap_train_df_2d, umap_test_df_2d, 
                    save_dir = '../result/23-07-21',
                    ds_name = 'FD003'):
    '''
    Ignore the time axis and perform UMAP to reduce the dimension to 3D and then visualize the working condition. 

    Parameters
    ----------
    umap_train_df_3d : DataFrame
        Train set in 3D low dimensional space.
    umap_test_df_3d : DataFrame
        Test set in 3D low dimensional space.
    save_dir : str, optional
        Directory to save the results. The default is '../result/23-07-21'.
    ds_name : str, optional
        The default is 'FD003'.

    Returns
    -------
    None.
    '''
    os.makedirs(save_dir, exist_ok=True)
    
    fig_train = px.scatter(umap_train_df_2d, x='y1', y='y2', 
                           color=umap_train_df_2d['WC']).update_traces(marker=dict(size=2))
    fig1 = go.Figure(data=fig_train.data, layout = fig_train.layout)
    fig1.update_layout(
            scene=dict(
                xaxis=dict(
                    title=' '* 30 +'y1',
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern')
                ),
                yaxis=dict(
                    title=' '* 30 +'y2',
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern')  
                )
            )
        )

    fig1.write_html(os.path.join(save_dir, 'WC-2d.html'))



def umap_plot_3d_WC(umap_train_df_3d, umap_test_df_3d, 
                    save_dir = '../result/23-07-21',
                    ds_name = 'FD003'):
    '''
    Ignore the time axis and perform UMAP to reduce the dimension to 3D and then visualize the working condition. 

    Parameters
    ----------
    umap_train_df_3d : DataFrame
        Train set in 3D low dimensional space.
    umap_test_df_3d : DataFrame
        Test set in 3D low dimensional space.
    save_dir : str, optional
        Directory to save the results. The default is '../result/23-07-21'.
    ds_name : str, optional
        The default is 'FD003'.

    Returns
    -------
    None.
    '''
    os.makedirs(save_dir, exist_ok=True)
    
    fig_train = px.scatter_3d(umap_train_df_3d, x='y1', y='y2', z='y3', 
                         color=umap_train_df_3d['WC'],
                         height=900, width=900).update_traces(marker=dict(size=2))
    fig1 = go.Figure(data=fig_train.data, layout = fig_train.layout)
    fig1.update_layout(
            scene=dict(
                xaxis=dict(
                    title=' '* 30 +'y1',
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern')
                ),
                yaxis=dict(
                    title=' '* 30 +'y2',
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern')  
                ),
                zaxis=dict(
                    title=' '* 30 +'y3',
                    titlefont=dict(size=32, family='Computer Modern'),
                    tickfont=dict(size=18, family='Computer Modern')
                )
            )
        )

    fig1.write_html(os.path.join(save_dir, 'WC.html'))



def robustness_umap():
    # robust test of UMAP, randomly choose some train units, and see the results. 
    ds_dir = '../dataset/Aircraft Engine/CMaps'    
    n_neighbors = 80
    min_dist = 1
    n_components = 2
    ds_name = 'FD003'
    train, test = prepare_data(ds_dir = ds_dir,
                               ds_name = ds_name, 
                               extract_rul_method = 'linear',
                               drop_useless = True,
                               drop_feature_lst = ['op1', 'op2', 'op3'])     

    start_time = time.time()
    unique_ids = train['id'].unique()
    n_units = len(unique_ids)
    for ratio in [0.3, 0.5, 0.7, 0.8, 0.85, 0.9]:
        n_units_to_discard = int(n_units * ratio)
        print(f'{n_units_to_discard = }')
        ids_to_discard = np.random.choice(unique_ids, n_units_to_discard, replace=False)
        train_drop = train[~train['id'].isin(ids_to_discard)].reset_index(drop=True)
        
        save_dir = f'../result-umap-0722-2024/{ds_name}-dropRatio-{ratio}-unsupervised-linearRUL-neighbors-{n_neighbors}-minDist-{min_dist}-{n_components}D'
        umap_train_df, umap_test_df = dr_umap(train_drop, test, ds_name,
                                              save_path = save_dir,
                                              normalize = True,
                                              drop_columns = ['id','cycle',
                                                              'op1','op2','op3','WC'],
                                              supervised = False,
                                              n_neighbors = n_neighbors, n_components = n_components, 
                                              metric='euclidean', min_dist=min_dist) 
        print(f"used_time: {time.time() - start_time} s") 
        
        umap_plot_2d_and_add_time(umap_train_df, umap_test_df, 
                                  save_dir = save_dir,
                                  ds_name = ds_name)


def umap_exp():
    '''
    grid search the best parameters.
    '''
    ds_dir = '../dataset/Aircraft Engine/CMaps'    
    n_components = 2
    for ds_name in ['FD001', 'FD002', 'FD003', 'FD004']:
        for n_neighbors in [15, 80, 100]:
            for min_dist in [0.1, 0.5, 1]:
                save_dir = f'../result-umap-0623-2024/{ds_name}-unsupervised-linearRUL-neighbors-{n_neighbors}-minDist-{min_dist}-{n_components}D'
                # load data
                train, test = prepare_data(ds_dir = ds_dir,
                                            ds_name = ds_name, 
                                            extract_rul_method = 'linear',
                                            drop_useless = True,
                                            drop_feature_lst = ['op1', 'op2', 'op3'])                        
                start_time = time.time()
                # perform UMAP
                umap_train_df, umap_test_df = dr_umap(train, test, ds_name,
                                                      save_path = save_dir,
                                                      normalize = True,
                                                      drop_columns = ['id','cycle',
                                                                      'op1','op2','op3','WC'],
                                                      supervised = False,
                                                      n_neighbors = n_neighbors, n_components = n_components, 
                                                      metric='euclidean', min_dist=min_dist) 
                print(f"used_time: {time.time() - start_time} s") # FD001: 287.35s; FD002: 715.91s; FD003: 342.58s; FD004: 844.91s
                umap_plot_2d_and_add_time(umap_train_df, umap_test_df, 
                                          save_dir = save_dir,
                                          ds_name = ds_name)
                umap_plot_2d_WC(umap_train_df, umap_test_df, 
                                    save_dir = save_dir,
                                    ds_name = ds_name)
                
    
    # plot 3d scatter plot with working conditions and RUL
    n_neighbors = 80
    min_dist = 1
    for ds_name in ['FD002','FD004']:
        n_components = 3
        save_dir = f'../result-umap/{ds_name}-unsupervised-linearRUL-neighbors-{n_neighbors}-minDist-{min_dist}-{n_components}D'
        umap_train_df = pd.read_csv(os.path.join(save_dir, f'{ds_name}-3d-train.csv'))
        umap_test_df = pd.read_csv(os.path.join(save_dir, f'{ds_name}-3d-test.csv'))
        umap_plot_3d_WC(umap_train_df, umap_test_df, 
                        save_dir = save_dir,
                        ds_name = ds_name)
        umap_plot_3d_RUL(umap_train_df, umap_test_df, 
                        save_dir = save_dir,
                        ds_name = ds_name)

    


def umap_FD004():
    ds_dir = '../dataset/Aircraft Engine/CMaps'    
    n_neighbors = 80
    min_dist = 1
    n_components = 2
    ds_name = 'FD004'
    train, test = prepare_data(ds_dir = ds_dir,
                               ds_name = ds_name, 
                               extract_rul_method = 'linear',
                               drop_useless = True,
                               drop_feature_lst = [])        
    
    
    # perform UMAP for each working condition
    for WC in train['WC'].unique():
        save_dir = f'../result-umap-0730-2024/WC_{WC}-{ds_name}-unsupervised-linearRUL-neighbors-{n_neighbors}-minDist-{min_dist}-{n_components}D'
        train_sub = train.query(f'WC == {WC}').reset_index(drop=True)
        test_sub = test.query(f'WC == {WC}').reset_index(drop=True)

        umap_train_df, umap_test_df = dr_umap(train_sub, test_sub, ds_name,
                                              save_path = save_dir,
                                              normalize = True,
                                              drop_columns = ['id','cycle',
                                                              'op1','op2','op3','WC'],
                                              supervised = False,
                                              n_neighbors = n_neighbors, n_components = n_components, 
                                              metric='euclidean', min_dist=min_dist) 

        umap_plot_2d_and_add_time(umap_train_df, umap_test_df, 
                                  save_dir = save_dir,
                                  ds_name = ds_name)

    


if __name__ == "__main__":    
    umap_exp()
    robustness_umap()
    umap_FD004()

    


    
    
    






















