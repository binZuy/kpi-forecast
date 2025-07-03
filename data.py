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

        # Chuẩn hóa target một cách cẩn thận
        target_col = target_df.columns[0]
        target_values = target_df[target_col].values
        
        # Kiểm tra và xử lý giá trị bất thường
        if np.any(np.isnan(target_values)) or np.any(np.isinf(target_values)):
            print("WARNING: Target contains NaN/Inf values, replacing with median")
            median_val = np.nanmedian(target_values)
            target_values = np.nan_to_num(target_values, nan=median_val, posinf=median_val, neginf=median_val)
            target_df = target_df.copy()
            target_df[target_col] = target_values
        
        # Chuẩn hóa target về khoảng [-1, 1] thay vì z-score
        target_min = target_values.min()
        target_max = target_values.max()
        if target_max > target_min:
            target_df = target_df.copy()
            target_df[target_col] = 2 * (target_values - target_min) / (target_max - target_min) - 1
        else:
            target_df = target_df.copy()
            target_df[target_col] = 0.0  # Nếu tất cả giá trị bằng nhau

        # QUAN TRỌNG: Tiền xử lý covariates trước khi sử dụng
        print("Tiền xử lý covariates...")
        covariate_df_processed = preprocess_full_dataframe(covariate_df)
        print(f"Covariates sau khi xử lý - shape: {covariate_df_processed.shape}")
        print(f"Covariates columns: {covariate_df_processed.columns.tolist()}")
        
        # Debug: Kiểm tra giá trị của các cột thời gian
        time_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear']
        for col in time_cols:
            if col in covariate_df_processed.columns:
                print(f"{col} min/max: {covariate_df_processed[col].min():.4f}/{covariate_df_processed[col].max():.4f}")

        target_arr = target_df.values  # [N, 1]
        covariate_arr = covariate_df_processed.values  # [N, num_features]

        # Kiểm tra covariates
        if np.any(np.isnan(covariate_arr)) or np.any(np.isinf(covariate_arr)):
            print("WARNING: Covariates contain NaN/Inf values, replacing with 0")
            covariate_arr = np.nan_to_num(covariate_arr, nan=0.0, posinf=0.0, neginf=0.0)

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
        
        # Debug: Kiểm tra scale của dữ liệu
        print(f"=== Debug MQRNN_Dataset ===")
        print(f"Target min/max: {self.targets.min():.4f}/{self.targets.max():.4f}")
        print(f"Target std: {self.targets.std():.4f}")
        print(f"Past covariates min/max: {self.past_covariates.min():.4f}/{self.past_covariates.max():.4f}")
        print(f"Past covariates std: {self.past_covariates.std():.4f}")
        print(f"Future covariates min/max: {self.future_covariates.min():.4f}/{self.future_covariates.max():.4f}")
        print(f"Future covariates std: {self.future_covariates.std():.4f}")
        
        # Kiểm tra xem có giá trị quá lớn không
        if self.future_covariates.max() > 5:
            print(f"WARNING: Future covariates có giá trị lớn: {self.future_covariates.max():.4f}")
        if self.targets.max() > 5:
            print(f"WARNING: Targets có giá trị lớn: {self.targets.max():.4f}")
        
        # Kiểm tra cụ thể các cột thời gian
        time_col_indices = []
        for i, col in enumerate(covariate_df_processed.columns):
            if col in ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear']:
                time_col_indices.append(i)
        
        if time_col_indices:
            print("Kiểm tra các cột thời gian:")
            for idx in time_col_indices:
                col_name = covariate_df_processed.columns[idx]
                col_values = self.future_covariates[:, :, idx]
                print(f"  {col_name}: min={col_values.min():.4f}, max={col_values.max():.4f}")
        
        print("=" * 30)

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

    # Kiểm tra xem target_col có tồn tại không
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    # Kiểm tra xem tất cả covariate_cols có tồn tại không
    missing_cols = [col for col in covariate_cols if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        # Chỉ giữ lại các cột có sẵn
        covariate_cols = [col for col in covariate_cols if col in df.columns]

    # Tách target và covariates
    target_df = df[[target_col]]
    covariate_df = df[covariate_cols]

    # Đảm bảo index là Date và được sắp xếp
    target_df = target_df.sort_index()
    covariate_df = covariate_df.sort_index()
    
    print(f"create_mqrnn_dataset:")
    print(f"  Target shape: {target_df.shape}")
    print(f"  Covariates shape: {covariate_df.shape}")
    print(f"  Covariates columns: {covariate_df.columns.tolist()}")
    
    # Kiểm tra giá trị của các cột thời gian
    time_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear']
    for col in time_cols:
        if col in covariate_df.columns:
            print(f"  {col} range: {covariate_df[col].min():.4f} - {covariate_df[col].max():.4f}")

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

    # Xử lý các cột thời gian - chuẩn hóa về khoảng [0, 1]
    time_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear']
    for col in time_cols:
        if col in df.columns:
            if col == 'Year':
                # Chuẩn hóa Year về khoảng [0, 1] với min=2013, max=2015
                df[col] = (df[col] - 2013) / (2015 - 2013)
            elif col == 'Month':
                # Chuẩn hóa Month về khoảng [0, 1]
                df[col] = (df[col] - 1) / 11  # 1-12 -> 0-1
            elif col == 'Day':
                # Chuẩn hóa Day về khoảng [0, 1]
                df[col] = (df[col] - 1) / 30  # 1-31 -> 0-1
            elif col == 'DayOfWeek':
                # Chuẩn hóa DayOfWeek về khoảng [0, 1]
                df[col] = (df[col] - 1) / 6  # 1-7 -> 0-1
            elif col == 'WeekOfYear':
                # Chuẩn hóa WeekOfYear về khoảng [0, 1]
                df[col] = (df[col] - 1) / 52  # 1-53 -> 0-1

    # Chuẩn hóa các cột numeric khác trừ boolean và one-hot encoded
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    one_hot_cols = [col for col in df.columns if any(prefix in col for prefix in ['StoreType_', 'Assortment_', 'PromoInterval_', 'StateHoliday_'])]
    
    for col in numeric_cols:
        if col not in boolean_cols and col not in one_hot_cols and col not in time_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0.0  # Nếu std = 0, set về 0

    # Đảm bảo tất cả các cột là float32
    df = df.astype(np.float32)

    return df

def debug_dataset_info(dataset: MQRNN_Dataset):
    """
    Debug thông tin về dataset để kiểm tra dữ liệu trước khi training
    """
    print("\n=== DEBUG DATASET INFO ===")
    print(f"Dataset length: {len(dataset)}")
    
    # Lấy sample đầu tiên
    encoder_input, future_covariate, target = dataset[0]
    
    print(f"Encoder input shape: {encoder_input.shape}")
    print(f"Encoder input dtype: {encoder_input.dtype}")
    print(f"Encoder input min/max: {encoder_input.min().item():.4f}/{encoder_input.max().item():.4f}")
    print(f"Encoder input std: {encoder_input.std().item():.4f}")
    
    print(f"Future covariate shape: {future_covariate.shape}")
    print(f"Future covariate dtype: {future_covariate.dtype}")
    print(f"Future covariate min/max: {future_covariate.min().item():.4f}/{future_covariate.max().item():.4f}")
    print(f"Future covariate std: {future_covariate.std().item():.4f}")
    
    print(f"Target shape: {target.shape}")
    print(f"Target dtype: {target.dtype}")
    print(f"Target min/max: {target.min().item():.4f}/{target.max().item():.4f}")
    print(f"Target std: {target.std().item():.4f}")
    
    # Kiểm tra xem có giá trị quá lớn không
    if future_covariate.max().item() > 10:
        print(f"WARNING: Future covariate có giá trị lớn: {future_covariate.max().item():.4f}")
    
    if target.max().item() > 10:
        print(f"WARNING: Target có giá trị lớn: {target.max().item():.4f}")
    
    print("=" * 30)
