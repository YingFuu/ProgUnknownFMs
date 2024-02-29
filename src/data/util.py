# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 14:59:20 2022

@author: Ying Fu
"""
import json
import os

import numpy as np
import pandas as pd

def find_useless_colum(df, max_missing_ratio=0.5, min_rows_per_value=2, 
                       max_ratio_per_cat=0.9, 
                       verbose=False, low_std_value = 0.01):
    '''
    Identify useless columns from a DataFrame. 
    Columns are divided into three types:
        num_float: contains numeric values only and at least a number that is not integer
        num_int:   contains numeric values and all values are integers (3.0 is treated as an integer)
        cat_like:  none numeric values are considered like categorical            
    
    If column is considered useless and classified into the following types:
        empty:              if the column contains no value
        singel-valued:      if the column contains only one value
        id-like:            A num_int or cat_like column contains a unique
                            value for each sample. It is okay for a num_float
                            column to contain a unqiue value for each sample
        too-many-missing:   if a column contains too many missing values --
                            exceeding total number of samples * max_missing_ratio
        too-small-cat:      if average samples per category are too few in a
                            cat_like column -- less than min_rows_per_value
        too-large-cat:      if a single category in a cat-like column contains
                            too many samples -- exceeding total number of
                            samples * max_ratio_per_cat

    Parameters
    ----------
    df : pandas.DataFrame
        A table contains many columns
    max_missing_ratio : float in [0.0,1.0], optional
        Threshold for determining a column to be too-many-missing. The default is 0.5.
    min_rows_per_value : int, optional
        Threshold for determining a column to be too-small-cat. The default is 2.
    max_ratio_per_cat : float in [0.0,1.0], optional
        Threshold for determining a column to be too-large-cat. The default is 0.9.
    verbose : bool, optional
        If True print more messages. The default is False.

    Returns
    -------
    dict
        A dictionary where a key represents a type of useless columns and
        the value is a list of useless columns of the corresponding type.

    '''
    empty_cols = []
    single_value_cols = []
    low_std_cols = []
    id_like_cols = []
    too_many_missing_cols = []
    too_small_cat_cols = []
    too_large_cat_cols = []

    # TODO: one-to-one map (two columns are one-to-one map), e.g. Crime 'AREA' and 'AREA NAME'
    # TODO: nearly one-to-one map  e.g. 'Premis Cd', 'Premis Desc' (both 803 and 805 are mapped to 'RETIRED (DUPLICATE) DO NOT USE THIS CODE'), rest is one-to-one)    
    # TODO: nearly one-to-one map  e.g. Crime 'Weapon Used Cd', 'Weapon Desc', 222 -> np.nan. The rest is one-to-one
    row_count = df.shape[0]
    for col in df:
        missing_count = df[col].isna().sum()
        if missing_count == row_count:
            if verbose:
                print(f'{col=} contains no value.')
            empty_cols.append(col)
            continue

        vc = df[col].value_counts(sort=True,dropna=True)
        if vc.size == 1:
            if missing_count == 0:
                if verbose:
                    print(f'{col=} contains a single value: {vc.index[0]}')
            else:
                if verbose:
                    print(f'{col=} contains a single value and missing value: {vc.index[0]}')
            single_value_cols.append(col)
            continue
        
        if df[col].std()< low_std_value:
            low_std_cols.append(col)
            continue
        
        # Cannot convert non-finite values (NA or inf) to integer
        inf_dropped = df.replace([np.inf, -np.inf], np.nan, inplace=False)
        na_dropped = inf_dropped[col].dropna()
        if not pd.api.types.is_numeric_dtype(na_dropped):
            col_type = 'cat_like'
        elif np.array_equal(na_dropped, na_dropped.astype(int)):
            col_type = 'num_int'
        else:
            col_type = 'num_float'
            
        # a unique value for each record
        if vc.size == row_count and col_type != 'num_float': 
            if col_type == 'cat_like':
                if verbose:
                    print(f'cat_like column: {col} has unique value for each row')    
                id_like_cols.append(col)
                continue
            else: # col_type == 'num_int'
                print(f'warning: int column: {col} has unique value for each row.')
        
        # a unique value for each record that has value
        if vc.size + missing_count == row_count and col_type != 'num_float': 
            if col_type == 'cat_like':
                if verbose:
                    print(f'cat_like column: {col} has unique value for each row that has value')    
                id_like_cols.append(col)
                continue
            else: # col_type == 'num_int'
                if verbose:
                    print(f'warning: int column: {col} has unique value for each row that has value')
        
        # missing rate exceed max_missing_ratio
        missing_count = df[col].isna().sum()
        if missing_count > max_missing_ratio * row_count:
            if verbose:
                print(f'{col=} has too many missing values: {missing_count}, missing ratio > {max_missing_ratio=}')
            too_many_missing_cols.append(col)
            continue

        # too few records per category
        if vc.size > 0:
            rows_per_value = row_count / vc.size
        else:
            rows_per_value = 0
        if rows_per_value < min_rows_per_value and col_type != 'num_float':
            if col_type == 'cat_like':
                if verbose:
                    print(f'cat_like column: {col} rows per cat: {rows_per_value} < {min_rows_per_value=}')
                too_small_cat_cols.append(col)
                continue
            else: # col_type == 'num_int':
                if verbose:
                    print(f'warning: int column: {col} rows per cat: {rows_per_value} < {min_rows_per_value=}')
        
        max_rows_per_cat = row_count * max_ratio_per_cat
        if vc.size > 0 and vc.iloc[0] > max_rows_per_cat:
            if col_type == 'cat_like':
                if verbose:
                    print(f'cat_like column: {col} rows for largest cat {vc.index[0]}: {vc.iloc[0]} > {max_ratio_per_cat=}')
                too_large_cat_cols.append(col)
                continue
            else: # col_type == 'num_int':
                if verbose:
                    print(f'warning: int column: {col} rows for largest cat {vc.index[0]}: {vc.iloc[0]} > {max_ratio_per_cat=}')
    
    return {'empty_cols': empty_cols,
            'single_value_cols': single_value_cols,
            'low_std_cols': low_std_cols,
            'id_like_cols': id_like_cols,
            'too_many_missing_cols': too_many_missing_cols,
            'too_small_cat_cols': too_small_cat_cols,
            'too_large_cat_cols': too_large_cat_cols}


def drop_useless(df, useless_cols_dict, verbose=True):
    '''
    Drop useless columns identified by find_useless_colum(df) from a dataframe df:
        drop(df, find_useless_colum(df))

    Parameters
    ----------
    df : pandas.DataFrame
        A data table.
    useless_cols_dict : dict(type,list)
        Use less columns identified by find_useless_colum(df,...) 
    verbose : bool, optional
        If true print more messages. The default is True.

    Returns
    -------
    df : pandas.DataFrame
        A copy of df with use less columns dropped.

    '''
    for useless_type in useless_cols_dict:
        cols = useless_cols_dict[useless_type]
        if verbose:
            print(f'drop {useless_type}: {cols}')
        df = df.drop(columns=cols)
    return df

def is_one_to_one(df, col1, col2):
    '''
    Check if col1 and col2 is one-to-one mapping

    Parameters
    ----------
    df : pandas.DataFrame
        A table
    col1 : string
        Name of a column in df
    col2 : string
        Name of a column in df

    Returns
    -------
    pd.Series
        If col1 and col2 is one-to-one mapping, return a series where index is value in col1 and value is value in col2;
        None otherwise.

    '''
    dfu = df.drop_duplicates([col1, col2])
    a = dfu.groupby(col1)[col2].count()
    b = dfu.groupby(col2)[col1].count()
    if (a.max() == 1 and a.min() == 1 and
        b.max() == 1 and b.min() == 1):
        return pd.Series(dfu[col2].values, index=dfu[col1].values)
    return None


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


def save_json(object, save_json_path, save_file_name):
    '''
    save an object to json file
    :param object:
    :param save_json_path:
    :param save_file_name:
    '''
    cwd = os.getcwd()
    os.makedirs(os.path.join(cwd, save_json_path), exist_ok=True)

    with open(os.path.join(save_json_path, save_file_name), 'w') as f:
        json.dump(object.__dict__, f, indent = 4)

def open_json(save_json_path, save_file_name):
    '''
    load a json file
    :param save_json_path:
    :param save_file_name:
    :return:
    '''
    with open(os.path.join(save_json_path, save_file_name)) as f:
        return json.load(f)