import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import TimeSeriesSplit

class MQRNN_Dataset(Dataset):
    def __init__(self, X, y, covariates):
        """
        X: [num_samples, input_window, num_stores]
        y: [num_samples, output_window, num_stores]
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
        num_stores = self.X.shape[2]
        input_series = self.X[idx]  # [input_window, num_stores]
        
        # Lấy covariates cho input window
        input_covariates = self.covariates[idx:idx+input_window]  # [input_window, num_stores, num_covariates]
        
        # Đầu vào cho decoder: covariate tương lai
        output_window = self.y.shape[1]
        future_covariates = self.covariates[idx+input_window:idx+input_window+output_window]  # [output_window, num_stores, num_covariates]
        
        # Đầu ra: chuỗi sales tương lai
        target = self.y[idx]  # [output_window, num_stores]

        # Chuyển sang tensor với dtype float64
        input_series = torch.tensor(input_series, dtype=torch.float64)  # [input_window, num_stores]
        input_covariates = torch.tensor(input_covariates, dtype=torch.float64)  # [input_window, num_stores, num_covariates]
        future_covariates = torch.tensor(future_covariates, dtype=torch.float64)  # [output_window, num_stores, num_covariates]
        target = torch.tensor(target, dtype=torch.float64)  # [output_window, num_stores]

        # In ra shape để debug
        print("input_series shape:", input_series.shape)
        print("input_covariates shape:", input_covariates.shape)
        print("future_covariates shape:", future_covariates.shape)

        # Tính trung bình sales cho mỗi ngày
        input_series_mean = input_series.mean(dim=1, keepdim=True)  # [input_window, 1]
        
        # Tính trung bình covariates cho mỗi ngày
        input_covariate_mean = input_covariates.mean(dim=1)  # [input_window, num_covariates]
        future_covariate_mean = future_covariates.mean(dim=1)  # [output_window, num_covariates]

        # In ra shape sau khi tính trung bình
        print("input_series_mean shape:", input_series_mean.shape)
        print("input_covariate_mean shape:", input_covariate_mean.shape)
        print("future_covariate_mean shape:", future_covariate_mean.shape)

        # Ghép input_series và input_covariate cho encoder
        encoder_input = torch.cat([input_series_mean, input_covariate_mean], dim=1)  # [input_window, 1+num_covariates]
        print("encoder_input shape:", encoder_input.shape)

        # Reshape future_covariate để phù hợp với encoder output
        # Chuyển future_covariate thành 3D tensor [input_window, batch_size, num_covariates]
        future_covariate = future_covariate_mean.unsqueeze(0)  # [1, output_window, num_covariates]
        future_covariate = future_covariate.expand(input_window, -1, -1)  # [input_window, output_window, num_covariates]
        
        # Reshape để phù hợp với encoder output
        future_covariate = future_covariate.reshape(input_window, -1)  # [input_window, output_window * num_covariates]
        future_covariate = future_covariate.unsqueeze(1)  # [input_window, 1, output_window * num_covariates]
        print("future_covariate shape:", future_covariate.shape)

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
    
    # Reshape covariates array và chuyển sang float64
    covariates_array = covariate_df.values.reshape(num_dates, num_stores, num_covariates).astype(np.float64)
    
    # Reshape target series để phù hợp với cấu trúc [num_samples, input_window]
    target_array = target_series.values.astype(np.float64)
    
    return target_array, covariates_array

def prepare_data_for_training(train, test, config):
    """
    Chuẩn bị dữ liệu cho training với:
    - Training data: 1/1/2013 - 31/7/2015
    - Prediction period: 1/8/2015 - 17/9/2015 (45 ngày)
    """
    # Lọc dữ liệu theo thời gian
    train_start = '2013-01-01'
    train_end = '2015-07-31'
    test_start = '2015-08-01'
    test_end = '2015-09-17'
    
    train = train[(train['Date'] >= train_start) & (train['Date'] <= train_end)]
    test = test[(test['Date'] >= test_start) & (test['Date'] <= test_end)]

    # Chuẩn hóa Sales theo từng store
    train['Sales'] = train.groupby('Store')['Sales'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    # Tạo dataset cho MQRNN
    train_target, train_covariates = create_mqrnn_dataset(train)
    test_target, test_covariates = create_mqrnn_dataset(test)

    # Tách dữ liệu thành input và target
    input_window = config.get('input_window', 90)  # Tăng input window lên 90 ngày để nắm bắt pattern dài hạn
    output_window = config.get('output_window', 45)  # 45 ngày dự đoán
    
    # Tạo input và target sequences
    X = []
    y = []
    for i in range(len(train_target) - input_window - output_window + 1):
        # Lấy input sequence cho mỗi store
        input_seq = train_target[i:i+input_window]  # [input_window, num_stores]
        # Lấy target sequence cho mỗi store
        target_seq = train_target[i+input_window:i+input_window+output_window]  # [output_window, num_stores]
        
        X.append(input_seq)
        y.append(target_seq)
    
    X = np.array(X)  # [num_samples, input_window, num_stores]
    y = np.array(y)  # [num_samples, output_window, num_stores]

    # In ra shape và thông tin để debug
    print("Training period:", train_start, "to", train_end)
    print("Prediction period:", test_start, "to", test_end)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("train_covariates shape:", train_covariates.shape)
    print("Number of samples:", len(X))
    print("Input window size:", input_window)
    print("Output window size:", output_window)

    # Tạo MQRNN dataset
    full_dataset = MQRNN_Dataset(
        X=X,  # Input sequences [num_samples, input_window, num_stores]
        y=y,  # Target sequences [num_samples, output_window, num_stores]
        covariates=train_covariates  # Covariates [num_samples, num_stores, num_covariates]
    )

    # Chia dataset thành train và validation (80-20 split)
    train_size = int(0.8 * len(full_dataset))
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

def remove_outliers(df, column, n_std=3):
    mean = df[column].mean()
    std = df[column].std()
    df = df[abs(df[column] - mean) <= (n_std * std)]
    return df


