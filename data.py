import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import TimeSeriesSplit

class MQRNN_Dataset(Dataset):
    def __init__(self, target_df, covariate_df, context_size, horizon_size, device='cpu'):
        """
        target_df: DataFrame với 1 cột target, index là thời gian
        covariate_df: DataFrame các covariate, index là thời gian
        context_size: số bước quá khứ (input window)
        horizon_size: số bước dự báo (output window)
        device: thiết bị (cpu hoặc cuda)
        """
        self.context_size = context_size
        self.horizon_size = horizon_size 
        self.device = device

        target_arr = target_df.values  # [N, 1]
        covariate_arr = covariate_df.values  # [N, num_features]

        num_samples = len(target_df) - context_size - horizon_size + 1

        # Precompute toàn bộ sample
        self.inputs = np.zeros((num_samples, context_size, 1), dtype=np.float64)
        self.past_covariates = np.zeros((num_samples, context_size, covariate_arr.shape[1]), dtype=np.float64)
        self.future_covariates = np.zeros((num_samples, horizon_size, covariate_arr.shape[1]), dtype=np.float64)
        self.targets = np.zeros((num_samples, horizon_size), dtype=np.float64)

        for i in range(num_samples):
            self.inputs[i] = target_arr[i:i+context_size]
            self.past_covariates[i] = covariate_arr[i:i+context_size]
            self.future_covariates[i] = covariate_arr[i+context_size:i+context_size+horizon_size]
            self.targets[i] = target_arr[i+context_size:i+context_size+horizon_size, 0]

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        # Ghép input series và covariate cho encoder
        encoder_input = np.concatenate([self.inputs[idx], self.past_covariates[idx]], axis=1)  # [context_size, 1+num_features]
        future_covariate = self.future_covariates[idx]  # [horizon_size, num_features]
        target = self.targets[idx]  # [horizon_size]
        return (
            torch.tensor(encoder_input, dtype=torch.float64, device=self.device),
            torch.tensor(future_covariate, dtype=torch.float64, device=self.device),
            torch.tensor(target, dtype=torch.float64, device=self.device)
        )

# class MQRNN_Dataset(torch.utils.data.Dataset):
#     def __init__(self,
#                 target_df: pd.DataFrame,
#                 covariate_df: pd.DataFrame,
#                 horizon_size: int,
#                 quantile_size: int,
#                 context_size: int):
#         """
#         Parameters:
#         -----------
#         target_df: pd.DataFrame
#             DataFrame chứa giá trị target (Sales) với index là Date
#         covariate_df: pd.DataFrame
#             DataFrame chứa các covariates với index là Date
#         horizon_size: int
#             Kích thước cửa sổ dự báo
#         quantile_size: int
#             Số lượng quantiles cần dự báo
#         """
#         print("\n=== Khởi tạo MQRNN_Dataset ===")
#         print(f"Shape của target_df: {target_df.shape}")
#         print(f"Shape của covariate_df: {covariate_df.shape}")
#         print(f"horizon_size: {horizon_size}")
#         print(f"quantile_size: {quantile_size}")

#         self.series_df = target_df
#         self.covariate_df = covariate_df.copy()
        
#         # Xử lý các cột categorical
#         self._process_categorical_columns()
        
#         self.horizon_size = horizon_size
#         self.quantile_size = quantile_size
#         self.context_size = context_size

#         # Calculate the number of possible sequences
#         self.seq_len = self.series_df.shape[0] - self.context_size - self.horizon_size
#         print(f"Số lượng mẫu có thể có: {self.seq_len}")
#         print(f"Số lượng features sau khi xử lý: {self.covariate_df.shape[1]}")

#         self.covariate_size = self.covariate_df.shape[1]
#         print(f"Số lượng covariates: {self.covariate_size}")

#         print("=== Hoàn thành khởi tạo ===")
    
#     def _process_categorical_columns(self):
#         """Xử lý các cột categorical trong covariates"""
#         print("\nĐang xử lý các cột categorical...")
        
#         # Xử lý StoreType
#         if 'StoreType' in self.covariate_df.columns:
#             print("Xử lý StoreType...")
#             # Chuyển đổi string thành số
#             store_type_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
#             self.covariate_df['StoreType'] = self.covariate_df['StoreType'].map(store_type_map)
#             # One-hot encoding
#             store_type_dummies = pd.get_dummies(self.covariate_df['StoreType'], prefix='StoreType')
#             self.covariate_df = pd.concat([self.covariate_df.drop('StoreType', axis=1), store_type_dummies], axis=1)
#             print(f"Số cột sau khi xử lý StoreType: {self.covariate_df.shape[1]}")
        
#         # Xử lý Assortment
#         if 'Assortment' in self.covariate_df.columns:
#             print("Xử lý Assortment...")
#             # Chuyển đổi string thành số
#             assortment_map = {'a': 0, 'b': 1, 'c': 2}
#             self.covariate_df['Assortment'] = self.covariate_df['Assortment'].map(assortment_map)
#             # One-hot encoding
#             assortment_dummies = pd.get_dummies(self.covariate_df['Assortment'], prefix='Assortment')
#             self.covariate_df = pd.concat([self.covariate_df.drop('Assortment', axis=1), assortment_dummies], axis=1)
#             print(f"Số cột sau khi xử lý Assortment: {self.covariate_df.shape[1]}")
        
#         # Xử lý PromoInterval
#         if 'PromoInterval' in self.covariate_df.columns:
#             print("Xử lý PromoInterval...")
#             # Chuyển đổi string thành số
#             promo_interval_map = {
#                 'Jan,Apr,Jul,Oct': 0,
#                 'Feb,May,Aug,Nov': 1,
#                 'Mar,Jun,Sept,Dec': 2
#             }
#             self.covariate_df['PromoInterval'] = self.covariate_df['PromoInterval'].map(promo_interval_map)
#             # One-hot encoding
#             promo_interval_dummies = pd.get_dummies(self.covariate_df['PromoInterval'], prefix='PromoInterval')
#             self.covariate_df = pd.concat([self.covariate_df.drop('PromoInterval', axis=1), promo_interval_dummies], axis=1)
#             print(f"Số cột sau khi xử lý PromoInterval: {self.covariate_df.shape[1]}")
        
#         # Xử lý StateHoliday
#         if 'StateHoliday' in self.covariate_df.columns:
#             print("Xử lý StateHoliday...")
#             # Chuyển đổi string thành số
#             state_holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
#             self.covariate_df['StateHoliday'] = self.covariate_df['StateHoliday'].map(state_holiday_map)
#             # One-hot encoding
#             state_holiday_dummies = pd.get_dummies(self.covariate_df['StateHoliday'], prefix='StateHoliday')
#             self.covariate_df = pd.concat([self.covariate_df.drop('StateHoliday', axis=1), state_holiday_dummies], axis=1)
#             print(f"Số cột sau khi xử lý StateHoliday: {self.covariate_df.shape[1]}")
        
#         # Xử lý các cột boolean
#         boolean_cols = ['Promo', 'Open', 'SchoolHoliday']
#         for col in boolean_cols:
#             if col in self.covariate_df.columns:
#                 self.covariate_df[col] = self.covariate_df[col].astype(int)
        
#         # Chuẩn hóa các cột số
#         numeric_cols = self.covariate_df.select_dtypes(include=[np.number]).columns
#         for col in numeric_cols:
#             if col not in ['Promo', 'Open', 'SchoolHoliday']:
#                 mean = self.covariate_df[col].mean()
#                 std = self.covariate_df[col].std()
#                 if std != 0:
#                     self.covariate_df[col] = (self.covariate_df[col] - mean) / std
        
#         # Chuyển đổi tất cả các cột sang float64
#         self.covariate_df = self.covariate_df.astype(np.float64)
        
#         print("\nCác cột sau khi xử lý:", self.covariate_df.columns.tolist())
#         print(f"Số lượng features: {self.covariate_df.shape[1]}")
#         print(f"self.covariate_df.shape[0] : {self.covariate_df.shape[0]}")
#     def __len__(self):
#         # The number of items is the number of possible sequences
#         return self.seq_len

#     def __getitem__(self, idx):
#         print(f"\n=== Xử lý item {idx} ===")
        
#         # Kiểm tra index hợp lệ
#         if idx < 0 or idx >= self.seq_len:
#             raise IndexError(f"Index {idx} nằm ngoài phạm vi [0, {self.seq_len-1}]")
        
#         # Lấy chuỗi thời gian hiện tại (input cho encoder)
#         cur_series = self.series_df.iloc[idx:idx+self.context_size, 0].values.astype(np.float64)
        
#         # Lấy covariates cho encoder
#         cur_covariate = self.covariate_df.iloc[idx:idx+self.context_size, :].values.astype(np.float64)
        
#         # Lấy covariates cho decoder (tương lai)
#         next_covariate = self.covariate_df.iloc[idx+self.context_size:idx+self.context_size+self.horizon_size, :].values.astype(np.float64)
#         next_covariate_tensor = torch.tensor(next_covariate, dtype=torch.float64)  # [horizon_size, num_features]
        
#         # Lấy giá trị thực tế cho tương lai (target)
#         real_vals = self.series_df.iloc[idx+self.context_size:idx+self.context_size+self.horizon_size, 0].values.astype(np.float64)
        
#         # Chuyển đổi sang tensor
#         cur_series_tensor = torch.tensor(cur_series, dtype=torch.float64).unsqueeze(1)  # [context_size, 1]
#         cur_covariate_tensor = torch.tensor(cur_covariate, dtype=torch.float64)  # [context_size, num_features]
        
#         # Ghép series và covariates cho encoder
#         cur_series_covariate_tensor = torch.cat([cur_series_tensor, cur_covariate_tensor], dim=1)  # [context_size, 1+num_features]
        
#         cur_real_vals_tensor = torch.tensor(real_vals, dtype=torch.float64)  # [horizon_size]
        
#         return cur_series_covariate_tensor, next_covariate_tensor, cur_real_vals_tensor


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

def preprocess_full_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiền xử lý toàn bộ DataFrame: categorical, one-hot, boolean, chỉ chuẩn hóa các cột continuous.
    Trả về DataFrame đã xử lý, chỉ gồm các cột số sẵn sàng cho model.
    """
    df = df.copy()

    # Xử lý StoreType
    if 'StoreType' in df.columns:
        store_type_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        df['StoreType'] = df['StoreType'].map(store_type_map)
        store_type_dummies = pd.get_dummies(df['StoreType'], prefix='StoreType')
        df = pd.concat([df.drop('StoreType', axis=1), store_type_dummies], axis=1)

    # Xử lý Assortment
    if 'Assortment' in df.columns:
        assortment_map = {'a': 0, 'b': 1, 'c': 2}
        df['Assortment'] = df['Assortment'].map(assortment_map)
        assortment_dummies = pd.get_dummies(df['Assortment'], prefix='Assortment')
        df = pd.concat([df.drop('Assortment', axis=1), assortment_dummies], axis=1)

    # Xử lý PromoInterval
    if 'PromoInterval' in df.columns:
        promo_interval_map = {
            'Jan,Apr,Jul,Oct': 0,
            'Feb,May,Aug,Nov': 1,
            'Mar,Jun,Sept,Dec': 2
        }
        df['PromoInterval'] = df['PromoInterval'].map(promo_interval_map)
        promo_interval_dummies = pd.get_dummies(df['PromoInterval'], prefix='PromoInterval')
        df = pd.concat([df.drop('PromoInterval', axis=1), promo_interval_dummies], axis=1)

    # Xử lý StateHoliday
    if 'StateHoliday' in df.columns:
        state_holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
        df['StateHoliday'] = df['StateHoliday'].map(state_holiday_map)
        state_holiday_dummies = pd.get_dummies(df['StateHoliday'], prefix='StateHoliday')
        df = pd.concat([df.drop('StateHoliday', axis=1), state_holiday_dummies], axis=1)

    # Xử lý các cột boolean
    boolean_cols = ['Promo', 'Open', 'SchoolHoliday']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Chỉ chuẩn hóa các cột continuous
    continuous_cols = [
        'Sales', 'Customers', 'SalePerCustomer', 'CompetitionDistance',
        'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
        'Promo2SinceWeek', 'Promo2SinceYear'
    ]
    for col in continuous_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                df[col] = (df[col] - mean) / std

    # Đảm bảo tất cả các cột là float32
    df = df.astype(np.float32)

    return df
