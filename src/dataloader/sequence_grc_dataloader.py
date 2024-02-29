import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SequenceDataset_grc(Dataset):
    '''
    Sliding window for time series prediction
        sample: (window_size,features)
        label: float
        data_id: int, indicate this sample belongs to which engine
        ste
    '''

    def __init__(self, df, window_size, stride,
                 no_train_col=['time_id', 'cycle', 'op1', 'op2', 'op3', 
                               'fm_prob', 'fm_label'],
                 y_column=['RUL']):

        df = df.reset_index(drop=True)

        self.df = df
        self.window_size = window_size
        self.stride = stride  # window moving stride
        self.data_tuples = []
        self.dtype = torch.float
        self.y_column = y_column

        all_columns = list(df.columns)
        X_column = list(set(all_columns).difference(set(no_train_col + y_column)))
        print(f"{X_column = }")
        X = df[X_column]
        y = df[y_column]
        self.norm_col_X = list(set(X_column).difference(set(['id'])))
        self.n_features = len(self.norm_col_X)

        X_copy = X.copy()
        y_copy = y.copy()

        for data_id in X_copy['id'].unique():
            X_sub = X_copy[X_copy['id'] == data_id]
            X_sub_copy = X_sub.copy()
            y_sub = y_copy.iloc[X_sub_copy.index]
            y_sub_copy = y_sub.copy()

            X_sub_copy.drop('id', axis=1, inplace=True)
            X_sub_copy = X_sub_copy.reset_index(drop=True)
            y_sub_copy = y_sub_copy.reset_index(drop=True)

            # if len(data) % window_size != 0, just skip and do not pad 0
            idxs = [i for i in range(0, len(X_sub_copy) + 1 - self.window_size, self.stride)]
            for j in idxs:
                sample = X_sub_copy.iloc[j:j + self.window_size, :]
                sample_copy = sample.copy().reset_index(drop=True)
                sample0 = sample_copy.iloc[:-1, :].reset_index(drop=True)
                sample1 = sample_copy.iloc[1:, :].reset_index(drop=True)
                
                label0 = y_sub_copy.iloc[j + window_size - 2:j + window_size-1].reset_index(drop=True)                
                label1 = y_sub_copy.iloc[j + window_size - 1:j + window_size].reset_index(drop=True)
                
                data_tuple = (sample0, label0, sample1, label1, data_id)
                self.data_tuples.append(data_tuple)

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):  # get the samples when used rather than load them at once.

        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_tuple = self.data_tuples[idx]
        sample0 = data_tuple[0]        
        label0 = data_tuple[1]
        sample1 = data_tuple[2]        
        label1 = data_tuple[3]        
        data_id = int(data_tuple[4])

        # transform dataframe to tensor.
        sample0 = torch.tensor(sample0.values, dtype=self.dtype)
        sample1 = torch.tensor(sample1.values, dtype=self.dtype)

        label0 = torch.tensor(label0.values, dtype=self.dtype)
        label0 = label0.squeeze()
        
        label1 = torch.tensor(label1.values, dtype=self.dtype)
        label1 = label1.squeeze()
        
        data_id = torch.tensor(data_id)

        return sample0, label0, sample1, label1, data_id

