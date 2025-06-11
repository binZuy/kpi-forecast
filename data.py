import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import TimeSeriesSplit

class MQRNN_Dataset(Dataset):
    def __init__(self, X, y, covariates):
        """
        X: [num_samples, input_window]
        y: [num_samples, output_window]
        covariates: [num_samples, num_stores, num_covariates]
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
        input_covariate = self.covariates[idx]  # [num_stores, num_covariates]
        
        # Đầu vào cho decoder: covariate tương lai
        output_window = self.y.shape[1]
        future_covariate = self.covariates[idx]  # [num_stores, num_covariates]
        
        # Đầu ra: chuỗi sales tương lai
        target = self.y[idx]  # [output_window]

        # Chuyển sang tensor
        input_series = torch.tensor(input_series, dtype=torch.float32).unsqueeze(-1)  # [input_window, 1]
        input_covariate = torch.tensor(input_covariate, dtype=torch.float32)         # [num_stores, num_covariates]
        future_covariate = torch.tensor(future_covariate, dtype=torch.float32)       # [num_stores, num_covariates]
        target = torch.tensor(target, dtype=torch.float32)                           # [output_window]

        # Ghép input_series và input_covariate cho encoder
        encoder_input = torch.cat([input_series, input_covariate], dim=1)            # [input_window, 1+num_covariates]

        return encoder_input, future_covariate, target

def load_and_preprocess_data(data_path='./data/rossmann-store-sales/'):
    """
    Load và xử lý dữ liệu từ các file CSV
    """
    # Đọc dữ liệu
    train = pd.read_csv(f'{data_path}/train.csv', low_memory=False)
    test = pd.read_csv(f'{data_path}/test.csv', low_memory=False)
    store = pd.read_csv(f'{data_path}/store.csv', low_memory=False)

    # Xử lý missing values
    test.fillna(1, inplace=True)
    store.CompetitionDistance = store.CompetitionDistance.fillna(store.CompetitionDistance.median())
    store.fillna(0, inplace=True)

    # Merge với store data
    train = pd.merge(train, store, on='Store')
    test = pd.merge(test, store, on='Store')

    # Chuyển đổi Date thành datetime
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])

    # Tạo các feature thời gian
    for df in [train, test]:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

    # Chuẩn hóa các feature số
    numeric_features = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 
                       'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']

    for feature in numeric_features:
        mean = train[feature].mean()
        std = train[feature].std()
        train[feature] = (train[feature] - mean) / std
        test[feature] = (test[feature] - mean) / std

    # One-hot encoding cho categorical variables
    categorical_cols = ['StoreType', 'Assortment', 'StateHoliday']
    train = pd.get_dummies(train, columns=categorical_cols)
    test = pd.get_dummies(test, columns=categorical_cols)

    return train, test

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
    
    # Tạo covariate dataframe - xử lý từng covariate riêng biệt
    covariate_dfs = []
    for col in covariate_cols:
        # Đảm bảo cột là numeric
        if df[col].dtype == 'object':
            # Nếu là cột categorical, chuyển thành numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaN với giá trị trung bình
            df[col] = df[col].fillna(df[col].mean())
        
        cov_df = df.pivot(index='Date', columns='Store', values=col)
        covariate_dfs.append(cov_df)
    
    # Kết hợp tất cả covariates
    covariate_df = pd.concat(covariate_dfs, axis=1)
    
    # Reshape covariates để phù hợp với cấu trúc [num_samples, num_stores, num_covariates]
    num_stores = len(target_series.columns)
    num_dates = len(target_series.index)
    num_covariates = len(covariate_cols)
    
    # Reshape covariates array và chuyển sang float32
    covariates_array = covariate_df.values.reshape(num_dates, num_stores, num_covariates).astype(np.float32)
    
    return target_series, covariates_array

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
        X=train_target.values,  # train_target là DataFrame nên cần .values
        y=train_target.values,  # train_target là DataFrame nên cần .values
        covariates=train_covariates  # train_covariates đã là numpy array nên không cần .values
    )

    # Chia dataset thành train và validation
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, val_idx in tscv.split(full_dataset):
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

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

def remove_outliers(df, column, n_std=3):
    mean = df[column].mean()
    std = df[column].std()
    df = df[abs(df[column] - mean) <= (n_std * std)]
    return df


