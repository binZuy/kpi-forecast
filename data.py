import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import TimeSeriesSplit

class MQRNN_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                target_df: pd.DataFrame,
                covariate_df: pd.DataFrame,
                horizon_size: int,
                quantile_size: int,
                context_size: int):
        """
        Parameters:
        -----------
        target_df: pd.DataFrame
            DataFrame chứa giá trị target (Sales) với index là Date
        covariate_df: pd.DataFrame
            DataFrame chứa các covariates với index là Date
        horizon_size: int
            Kích thước cửa sổ dự báo
        quantile_size: int
            Số lượng quantiles cần dự báo
        """
        print("=== Khởi tạo MQRNN_Dataset ===")
        print(f"Shape của target_df: {target_df.shape}")
        print(f"Shape của covariate_df: {covariate_df.shape}")
        print(f"horizon_size: {horizon_size}")
        print(f"quantile_size: {quantile_size}")

        self.series_df = target_df
        self.covariate_df = covariate_df.copy()  # Tạo bản sao để tránh thay đổi dữ liệu gốc
        
        # Xử lý các cột categorical
        self._process_categorical_columns()
        
        self.horizon_size = horizon_size
        self.quantile_size = quantile_size
        self.context_size = context_size

        # Calculate the number of possible sequences
        self.seq_len = self.series_df.shape[0] - self.horizon_size
        print(f"Số lượng chuỗi dự đoán có thể có (seq_len): {self.seq_len}")

        self.covariate_size = self.covariate_df.shape[1]
        print(f"Số lượng covariates: {self.covariate_size}")

        print("=== Hoàn thành khởi tạo ===")
    
    def _process_categorical_columns(self):
        """Xử lý các cột categorical trong covariates"""
        print("Đang xử lý các cột categorical...")
        
        # Xử lý StoreType
        if 'StoreType' in self.covariate_df.columns:
            print("Xử lý StoreType...")
            store_type_dummies = pd.get_dummies(self.covariate_df['StoreType'], prefix='StoreType')
            self.covariate_df = pd.concat([self.covariate_df.drop('StoreType', axis=1), store_type_dummies], axis=1)
        
        # Xử lý Assortment
        if 'Assortment' in self.covariate_df.columns:
            print("Xử lý Assortment...")
            assortment_dummies = pd.get_dummies(self.covariate_df['Assortment'], prefix='Assortment')
            self.covariate_df = pd.concat([self.covariate_df.drop('Assortment', axis=1), assortment_dummies], axis=1)
        
        # Xử lý PromoInterval
        if 'PromoInterval' in self.covariate_df.columns:
            print("Xử lý PromoInterval...")
            # Tách các tháng và tạo one-hot encoding
            promo_intervals = self.covariate_df['PromoInterval'].str.split(',')
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
            for month in months:
                self.covariate_df[f'Promo_{month}'] = self.covariate_df['PromoInterval'].str.contains(month).astype(int)
            self.covariate_df = self.covariate_df.drop('PromoInterval', axis=1)
        
        # Xử lý các cột boolean
        boolean_cols = ['Promo', 'Open', 'SchoolHoliday']
        for col in boolean_cols:
            if col in self.covariate_df.columns:
                self.covariate_df[col] = self.covariate_df[col].astype(int)
        
        # Xử lý StateHoliday
        if 'StateHoliday' in self.covariate_df.columns:
            self.covariate_df['StateHoliday'] = self.covariate_df['StateHoliday'].map({
                '0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4
            })
        
        # Chuẩn hóa các cột số
        numeric_cols = self.covariate_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Promo', 'Open', 'SchoolHoliday', 'StateHoliday']:
                mean = self.covariate_df[col].mean()
                std = self.covariate_df[col].std()
                if std != 0:
                    self.covariate_df[col] = (self.covariate_df[col] - mean) / std
        
        # Chuyển đổi tất cả các cột sang float64
        self.covariate_df = self.covariate_df.astype(np.float64)
        print(f"Số lượng covariates sau khi xử lý: {self.covariate_df.shape[1]}")

    def __len__(self):
        # The number of items is the number of possible sequences
        return self.seq_len

    def __getitem__(self, idx):
        print(f"\n=== Xử lý item {idx} ===")
        
        # Kiểm tra index hợp lệ
        if idx < 0 or idx >= self.seq_len:
            raise IndexError(f"Index {idx} nằm ngoài phạm vi [0, {self.seq_len-1}]")
        
        # Lấy chuỗi thời gian hiện tại (input cho encoder)
        cur_series = self.series_df.iloc[idx:idx+self.context_size, 0].values.astype(np.float64)
        print(f"Shape của cur_series: {cur_series.shape}")
        
        # Lấy covariates cho encoder
        cur_covariate = self.covariate_df.iloc[idx:idx+self.context_size, :].values.astype(np.float64)
        print(f"Shape của cur_covariate: {cur_covariate.shape}")
        
        # Lấy covariates cho decoder (tương lai)
        next_covariate = self.covariate_df.iloc[idx+self.context_size:idx+self.context_size+self.horizon_size, :].values.astype(np.float64)
        print(f"Shape của next_covariate: {next_covariate.shape}")
        
        # Lấy giá trị thực tế cho tương lai (target)
        real_vals = self.series_df.iloc[idx+self.context_size:idx+self.context_size+self.horizon_size, 0].values.astype(np.float64)
        print(f"Shape của real_vals: {real_vals.shape}")
        
        # Chuyển đổi sang tensor
        cur_series_tensor = torch.tensor(cur_series, dtype=torch.float64).unsqueeze(1)  # [context_size, 1]
        cur_covariate_tensor = torch.tensor(cur_covariate, dtype=torch.float64)  # [context_size, num_features]
        
        # Ghép series và covariates cho encoder
        cur_series_covariate_tensor = torch.cat([cur_series_tensor, cur_covariate_tensor], dim=1)  # [context_size, 1+num_features]
        print(f"Shape của cur_series_covariate_tensor: {cur_series_covariate_tensor.shape}")
        
        # Thêm batch dimension
        cur_series_covariate_tensor = cur_series_covariate_tensor.unsqueeze(0)  # [1, context_size, 1+num_features]
        print(f"Shape của cur_series_covariate_tensor sau khi thêm batch: {cur_series_covariate_tensor.shape}")
        
        next_covariate_tensor = torch.tensor(next_covariate, dtype=torch.float64)  # [horizon_size, num_features]
        next_covariate_tensor = next_covariate_tensor.unsqueeze(0)  # [1, horizon_size, num_features]
        print(f"Shape của next_covariate_tensor: {next_covariate_tensor.shape}")
        
        cur_real_vals_tensor = torch.tensor(real_vals, dtype=torch.float64)  # [horizon_size]
        print(f"Shape của cur_real_vals_tensor: {cur_real_vals_tensor.shape}")
        
        return cur_series_covariate_tensor, next_covariate_tensor, cur_real_vals_tensor
    
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
    Tạo dataset cho MQRNN từ DataFrame, giữ nguyên cấu trúc dữ liệu gốc.
    """
    if covariate_cols is None:
        covariate_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear',
            'CompetitionDistance', 'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear',
            'Promo', 'StateHoliday', 'SchoolHoliday', 'Open', 'SalePerCustomer']
    
    # Tách target và covariates
    target_df = df[[target_col]]
    covariate_df = df[covariate_cols]
    
    # Đảm bảo index là Date và được sắp xếp
    target_df = target_df.sort_index()
    covariate_df = covariate_df.sort_index()
    
    return target_df, covariate_df

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
        df=train,  # DataFrame với cấu trúc: Date, Store, DayOfWeek, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday
        input_window=input_window,
        output_window=output_window
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

def prepare_data_for_mqrnn(df, target_col='Sales', covariate_cols=None):
    """
    Chuẩn bị dữ liệu cho MQRNN
    
    Parameters:
    -----------
    df: DataFrame
        DataFrame chứa dữ liệu
    target_col: str
        Tên cột chứa giá trị target (Sales)
    covariate_cols: list
        Danh sách các cột làm covariates
    """
    # Tạo dataset
    dataset = MQRNN_Dataset(df, target_col=target_col, covariate_cols=covariate_cols)
    
    # Tạo dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataset, dataloader

# Ví dụ sử dụng
if __name__ == "__main__":
    # Đọc dữ liệu
    df = pd.read_csv('data.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    
    # Chỉ định các cột covariates
    covariate_cols = ['DayOfWeek', 'Customers', 'Open', 'Promo', 
                      'StateHoliday', 'SchoolHoliday']
    
    # Chuẩn bị dữ liệu
    dataset, dataloader = prepare_data_for_mqrnn(df, target_col='Sales', 
                                               covariate_cols=covariate_cols)
    
    # Kiểm tra một batch
    for batch in dataloader:
        input_covariates, future_covariates, target = batch
        print("Input covariates shape:", input_covariates.shape)
        print("Future covariates shape:", future_covariates.shape)
        print("Target shape:", target.shape)
        break


