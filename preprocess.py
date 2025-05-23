import pandas as pd
import numpy as np

# Đọc dữ liệu
train = pd.read_csv('data/rossmann-store-sales/train.csv', low_memory=False)
test = pd.read_csv('data/rossmann-store-sales/test.csv', low_memory=False)
store = pd.read_csv('data/rossmann-store-sales/store.csv', low_memory=False)

# Merge store info
train = train.merge(store, on='Store', how='left')
test = test.merge(store, on='Store', how='left')

# Tạo feature thời gian
for df in [train, test]:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    # ... thêm các feature khác

# Xử lý missing value
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# Tạo window cho mỗi store
# (giả sử input_window=30, output_window=7)
input_window = 30
output_window = 7
X, y, covariates = [], [], []

feature_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear']  # các covariate sử dụng

for store_id, group in train.groupby('Store'):
    group = group.sort_values('Date')
    sales = group['Sales'].values
    covs = group[feature_cols].values

    # Chuẩn hóa sales theo từng store (nếu muốn)
    sales_mean = sales.mean()
    sales_std = sales.std() if sales.std() > 0 else 1
    sales = (sales - sales_mean) / sales_std

    for i in range(len(sales) - input_window - output_window + 1):
        X.append(sales[i:i+input_window])
        y.append(sales[i+input_window:i+input_window+output_window])
        covariates.append(covs[i:i+input_window+output_window])

X = np.array(X)  # [num_samples, input_window]
y = np.array(y)  # [num_samples, output_window]
covariates = np.array(covariates)  # [num_samples, input_window+output_window, num_covariates]

# Lưu dữ liệu và metadata
np.savez('processed_train.npz', 
         X=X, 
         y=y, 
         covariates=covariates,
         feature_names=feature_cols,
         input_window=input_window,
         output_window=output_window)
