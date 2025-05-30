import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class MQRNN_Dataset(Dataset):
    def __init__(self, X, y, covariates):
        """
        X: [num_samples, input_window]
        y: [num_samples, output_window]
        covariates: [num_samples, input_window+output_window, num_covariates]
        """
        self.X = X
        self.y = y
        self.covariates = covariates

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Đầu vào cho encoder: chuỗi sales quá khứ + covariate quá khứ
        input_window = self.X.shape[1]
        input_series = self.X[idx]  # [input_window]
        input_covariate = self.covariates[idx, :input_window, :]  # [input_window, num_covariates]
        # Đầu vào cho decoder: covariate tương lai
        output_window = self.y.shape[1]
        future_covariate = self.covariates[idx, input_window:, :]  # [output_window, num_covariates]
        # Đầu ra: chuỗi sales tương lai
        target = self.y[idx]  # [output_window]

        # Chuyển sang tensor
        input_series = torch.tensor(input_series, dtype=torch.float32).unsqueeze(-1)  # [input_window, 1]
        input_covariate = torch.tensor(input_covariate, dtype=torch.float32)         # [input_window, num_covariates]
        future_covariate = torch.tensor(future_covariate, dtype=torch.float32)       # [output_window, num_covariates]
        target = torch.tensor(target, dtype=torch.float32)                           # [output_window]

        # Ghép input_series và input_covariate cho encoder
        encoder_input = torch.cat([input_series, input_covariate], dim=1)            # [input_window, 1+num_covariates]

        return encoder_input, future_covariate, target

def create_mqrnn_dataset(df, target_col='Sales', covariate_cols=None):
    """
    Tạo dataset cho MQRNN từ DataFrame
    """
    if covariate_cols is None:
        covariate_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear',
                         'CompetitionDistance', 'CompetitionOpenSinceMonth',
                         'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear',
                         'Promo', 'StateHoliday', 'SchoolHoliday', 'Open']
    
    # Sắp xếp theo Store và Date
    df = df.sort_values(['Store', 'Date'])
    
    # Tạo target series
    target_series = df.pivot(index='Date', columns='Store', values=target_col)
    
    # Tạo covariate dataframe
    covariate_df = df.pivot(index='Date', columns='Store', values=covariate_cols)
    covariate_df.columns = [f"{col}_{store}" for col, store in covariate_df.columns]
    
    return target_series, covariate_df

def prepare_data_for_training(train, test, config):
    """
    Chuẩn bị dữ liệu cho training
    """
    # Chuẩn hóa Sales theo từng store
    train['Sales'] = train.groupby('Store')['Sales'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    # Tạo dataset cho MQRNN
    train_target, train_covariates = create_mqrnn_dataset(train)
    test_target, test_covariates = create_mqrnn_dataset(test)

    # Tạo MQRNN dataset
    full_dataset = MQRNN_Dataset(
        X=train_target.values,
        y=train_target.values,
        covariates=train_covariates.values
    )

    # Chia dataset thành train và validation
    train_size = int(config['train_ratio'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Tạo dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    return train_loader, val_loader, test_target, test_covariates

def get_feature_names():
    """
    Trả về danh sách tên các features
    """
    return ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear',
            'CompetitionDistance', 'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear',
            'Promo', 'StateHoliday', 'SchoolHoliday', 'Open']


