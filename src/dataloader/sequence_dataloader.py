import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    '''
    Sliding window for time series prediction
        sample: (window_size,features)
        label: float
        data_id: int, indicate this sample belongs to which engine
    '''

    def __init__(self, df, window_size, stride,
                 no_train_col = ['cycle', 'op1', 'op2', 'op3','fm_label','WC'],
                 y_column = ['RUL']):
        
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
                sample_copy = sample.copy()
                sample_copy = sample_copy.reset_index(drop=True)
                label = y_sub_copy.iloc[j + self.window_size - 1:j + self.window_size]
                label_copy = label.copy()
                label_copy = label_copy.reset_index(drop=True)
                data_tuple = (sample_copy, label_copy, data_id)
                self.data_tuples.append(data_tuple)

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):  # get the samples when used rather than load them at once.

        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_tuple = self.data_tuples[idx]
        sample = data_tuple[0]
        label = data_tuple[1]
        data_id = int(data_tuple[2])

        # transform dataframe to tensor.
        sample = torch.tensor(sample.values, dtype=self.dtype)
        label = torch.tensor(label.values, dtype=self.dtype)
        label = label.squeeze()
        data_id = torch.tensor(data_id)

        return sample, label, data_id

