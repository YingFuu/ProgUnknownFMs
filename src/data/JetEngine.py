# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 19:35:55 2022
Load and preprocess NASA Jet Engine Dataset
@author: Ying Fu
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Times New Roman'


if __name__ == "__main__":
    import util 
else:
    from . import util  # when it is a module and being imported. 


def read_data(ds_dir='../../dataset/Aircraft Engine/CMaps',
              file_name='train_FD001.txt'):
    '''
    read train or test set.
    
    Parameters
    ----------
    ds_dir : str, optional
        directory of the dataset. The default is '../dataset/Aircraft Engine/CMaps'.
    file_name : str, optional
        filename of the dataset. The default is 'train_FD001.txt'.

    Returns
    -------
    train_data : DataFrame
    '''

    columns = ['id', 'cycle', 'op1', 'op2', 'op3', 'T2', 'T24', 'T30',
               'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc',
               'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR',
               'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']

    df = pd.read_csv(os.path.join(ds_dir, file_name), 
                     sep='\s+', header=None, names=columns)

    for col in columns:
        df[col] = df[col].astype('float')
        
    return df


def read_ground_truth(ds_dir='../../dataset/Aircraft Engine/CMaps',
                      file_name = 'RUL_FD001.txt'):
    '''
    read true remaining cycles for each engine in the testing data
    
    Parameters
    ----------
    ds_dir : str, optional
        directory of the dataset. The default is '../dataset/Aircraft Engine/CMaps'.
    file_name : str, optional
        filename of the dataset. The default is 'RUL_FD001.txt'.

    Returns
    -------
    ground_truth_df : DataFrame, with shape (100,1)

    '''
    ground_truth_df = pd.read_csv(os.path.join(ds_dir, file_name), 
                           sep=' ', header=None)
    ground_truth_df.drop(ground_truth_df.columns[[1]], axis=1, inplace=True) 
    
    return ground_truth_df


def extract_RUL_train(df, method = 'piece-wise'):
    # Get the total number of cycles for each unit
    max_cycle = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
    max_cycle.columns = ['id', 'max_cycle']

    # Merge the max cycle back into the original frame
    df = df.merge(max_cycle, on=['id'], how='left') 
    
    if method == 'linear':
        df['RUL'] = df['max_cycle'] - df['cycle'] 
    elif method == "piece-wise":
        df['RUL'] = df['max_cycle'] - df['cycle'] 
        df['RUL'].where(df['RUL'] <= 125, 125, inplace=True) # Where cond is True, keep the original value. Where False, replace with corresponding value from other
    else:
        raise RuntimeError(f'{method} does not exist')
    
    # 'max_cycle' and 'cycle' is used for extracting RUL.
    # Once 'RUL' is obtained, drop the two columns 
    # df.drop(['max_cycle','cycle'], axis=1, inplace=True) 
    df.drop(['max_cycle'], axis=1, inplace=True) 
    
    return df


def extract_RUL_test(df, ds_dir='../../dataset/Aircraft Engine/CMaps', 
                     ds_name = 'FD001', 
                     method = 'piece-wise'):
    
    if ds_name != 'FD005':
        file_name = 'RUL_'+ds_name + '.txt'
        test_rul_df = pd.read_csv(os.path.join(ds_dir, file_name), 
                                  header=None)
        print(test_rul_df.head())
    if ds_name == 'FD005':
        data_file = os.path.join(ds_dir, 'Turbofan-Engine-Degradation-Dataset-with-Multiple-Failure-Modes')
        test_rul_df = pd.read_csv(os.path.join(data_file, 'multi_modes_rul.csv'), 
                                         header=None)
        
    test_rul_df.columns = ['ground_truth']
    sub_max_cycle = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
    sub_max_cycle.columns = ['id', 'sub_max_cycle']
    
    max_cycle = pd.concat([sub_max_cycle, test_rul_df], axis=1, join='inner')
    max_cycle["max_cycle"] = max_cycle['sub_max_cycle'] + max_cycle['ground_truth']
    max_cycle.drop(['ground_truth','sub_max_cycle'], axis=1, inplace=True)
    
    # Merge the max cycle back into the original frame
    df = df.merge(max_cycle, on=['id'], how='left') 
    
    if method == 'linear':
        df['RUL'] = df['max_cycle'] - df['cycle'] 
    elif method == "piece-wise":
        df['RUL'] = df['max_cycle'] - df['cycle'] 
        df['RUL'].where(df['RUL'] <= 125, 125, inplace=True) # Where cond is True, keep the original value. Where False, replace with corresponding value from other
    else:
        raise RuntimeError(f'{method} does not exist')
    
    # 'max_cycle' and 'cycle' is used for extracting RUL.
    # Once 'RUL' is obtained, drop the two columns 
    # df.drop(['max_cycle','cycle'], axis=1, inplace=True) 
    df.drop(['max_cycle'], axis=1, inplace=True) 
    
    return df



def add_working_condition(df):
    '''
    For datasets with multiple working conditions, add an additional column 'WC'
    
    Parameters
    ----------
    df : DataFrame. Without 'WC' column. 
    
    
    Returns
    -------
    df : DataFrame. With 'WC' column. 
    '''
    
    
    df['op1'] = df['op1'].apply(lambda x: round(x))
          
    df['WC'] = 0
    df.loc[df['op1']==0, 'WC'] = 0
    df.loc[df['op1']==10, 'WC'] = 1
    df.loc[df['op1']==20, 'WC'] = 2
    df.loc[df['op1']==25, 'WC'] = 3
    df.loc[df['op1']==35, 'WC'] = 4
    df.loc[df['op1']==42, 'WC'] = 5
    
    return df



def prepare_data(ds_dir='../../dataset/Aircraft Engine/CMaps',
                 ds_name = 'FD001', extract_rul_method ='piece-wise',
                 drop_useless = True, drop_feature_lst = []):
    
    if ds_name in ['FD001', 'FD002', 'FD003', 'FD004']:
        train_file_name = 'train_'+ ds_name + '.txt'
        test_file_name = 'test_'+ ds_name + '.txt'
        
        train_df = read_data(ds_dir = ds_dir, file_name= train_file_name)
        train_df = extract_RUL_train(train_df, method = extract_rul_method)
    
        test_df =  read_data(ds_dir = ds_dir, file_name= test_file_name)
        test_df = extract_RUL_test(test_df, ds_dir=ds_dir,
                                   ds_name = ds_name,
                                   method = extract_rul_method)
        
        train_df = add_working_condition(train_df)
        test_df = add_working_condition(test_df)
    
    elif ds_name in ['FD005']:
        data_file = os.path.join(ds_dir, 'Turbofan-Engine-Degradation-Dataset-with-Multiple-Failure-Modes')
        columns = ['mode', 'id', 'cycle',
                      'NL', 'NH', 'P13', 'P26','T26',
                      'P3','T3','T6','EPR','T13','P42',
                      'T42','P5','T41','Thrust','Wf']
        train_df = pd.read_csv(os.path.join(data_file, 'multi_mode_train_data.csv'), 
                                         header=None,names=columns)
        train_df = train_df.drop('mode', axis=1)
        train_df = extract_RUL_train(train_df, method = extract_rul_method)
        
        test_df = pd.read_csv(os.path.join(data_file, 'multi_modes_test_data.csv'), 
                                         header=None,names=columns)
        test_df = test_df.drop('mode', axis=1)
        test_df = extract_RUL_test(test_df, ds_dir=ds_dir, ds_name = ds_name, method = extract_rul_method)
        
        train_df['WC']=0
        test_df['WC']=0

    else:
        raise RuntimeError(f'{ds_name} do not exist')

    if len(drop_feature_lst)>=1:
        for col in drop_feature_lst:
            if col in list(train_df.columns):                
                train_df.drop(columns=col, inplace = True)
                test_df.drop(columns=col, inplace = True)
    
    train_df_temp = train_df.drop('WC', axis=1)
    if drop_useless:
        useless_cols_dict = util.find_useless_colum(train_df_temp)
        train_df = util.drop_useless(train_df, useless_cols_dict)
        test_df = util.drop_useless(test_df, useless_cols_dict)
    
    if train_df.shape[1] != test_df.shape[1]:
        raise RuntimeError('The number of features of train and test do not match')

    print(f'column names: {train_df.columns}')
    
    return train_df, test_df
    
   

################ Explore data ################
def explore_data(ds_list):
    '''
    1. for all engine, check the extreme sensors(fails at the maximum or minimal value)
    2. plot the standard deviation of each sensor
    3. plot the histgram of train_data at three different operational settings
    '''
    
    def extreme_sensor(train_data):
        '''
        for all engine, check the extreme sensors(fails at the maximum or minimal value)
        :param train_data: DataFrame, the input train data
        :return: extreme_sensor_dict, dictionary, records the count of the extreme sensors
        '''
        counts = len(train_data['id'].unique())
        print(f'{counts = }')
        extreme_sensor_dict = {}
        for id in train_data['id'].unique():
            df = train_data.query(f'id == {id}')
            for i in range(1,22):
                title = 'sensor' + str(i)
                max_v = df[title].max()
                min_v = df[title].min()
                if max_v == min_v:
                    continue
                if df[title].iloc[-1] == max_v:
                    # print(title + '          :max')
                    if title in extreme_sensor_dict:
                        extreme_sensor_dict[title] += 1
                    else:
                        extreme_sensor_dict[title] = 1
                if df[title].iloc[-1] == min_v:
                    # print(title + '          :min')
                    if title in extreme_sensor_dict:
                        extreme_sensor_dict[title] += 1
                    else:
                        extreme_sensor_dict[title] = 1
                        
        extreme_sensor_dict = {k: v for k, v in sorted(extreme_sensor_dict.items(), key=lambda item: item[1], reverse=True)}
        
        return extreme_sensor_dict


    def sensors_std(train_data):
        '''
        check the std 
        :param train_data:
        :return:
        '''
        stats = train_data.agg(['mean', 'std']).T[2:]
        stats.plot.bar(y='std')
        plt.title('sensor std')
        plt.xlabel('sensor')
        plt.ylabel('std')
        plt.show()


    def ops_settings_plot(train_data):
        '''
        plot the histgram of train_data at three different operational settings
        :param train_data:
        :return:
        '''
        plt.subplot(311)
        train_data['op1'].plot.hist(bins=50)
        plt.title('op1')

        plt.subplot(312)
        train_data['op2'].plot.hist(bins=50)
        plt.title('op2')

        plt.subplot(313)
        train_data['op3'].plot.hist(bins=50)
        plt.title('op3')

        plt.tight_layout()
        plt.show()

    for train_data in ds_list:
        ops_settings_plot(train_data)
        extreme_sensor_dict = extreme_sensor(train_data)
        print(f'{extreme_sensor_dict}')
        sensors_std(train_data, delta=None)
        
        
    def op_3d_plot(df):
        '''
        FD001 has 1 Operating Conditions and 1 Fault Modes.
        FD002 has 6 Operating Conditions and 1 Fault Modes.    
        FD003 has 1 Operating Conditions and 2 Fault Modes.
        FD004 has 6 Operating Conditions and 2 Fault Modes.
        There are total 3 operations settings in the datasets, acutually, these 3 operational settings 
        are the combinations of true Operating Conditions. 
        Plot the 3 operations settings in a 3D graph to verify it.     
        
        -------------
        6 conditions:
            Operation        Altitude(Kft)        Mach number        TRA
            1                35                   0.8400             100
            2                42                   0.8408             100
            3                25                   0.6218             60
            4                25                   0.7002             100
            5                20                   0.2516             100 
            6                10                   0.7002             100
        '''

        ax = plt.axes(projection='3d')
        ax.scatter3D(df['op1'], 
                     df['op2'],
                     df['op3'])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        plt.show()
  
    
if __name__ == '__main__':
    # df_01 = read_data(ds_dir='../../dataset/Aircraft Engine/CMaps',
    #           file_name='train_FD001.txt')
    # s1 = df_01.describe()
    # df_01_unique = df_01.nunique()
    #
    # df_02 = read_data(ds_dir='../../dataset/Aircraft Engine/CMaps',
    #           file_name='train_FD002.txt')
    # s2 = df_02.describe()
    # df_02_unique = df_02.nunique()
    #
    # df_03 = read_data(ds_dir='../../dataset/Aircraft Engine/CMaps',
    #           file_name='train_FD003.txt')
    # s3 = df_03.describe()
    # df_03_unique = df_03.nunique()

    df_04 = read_data(ds_dir='../../dataset/Aircraft Engine/CMaps',
              file_name='train_FD004.txt')
    df_04_sub = df_04.loc[(df_04['op1'] > 40) & (df_04['op2'] > 0.84) & (df_04['op3'] ==100 )]
    s4 = df_04.describe()
    df_04_unique = df_04.nunique()








