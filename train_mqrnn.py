import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from MQRNN import Encoder, Decoder
from MQRNN.MQRNN import MQRNN
from MQRNN.data import MQRNN_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Thiết lập config
config = {
    'horizon_size': 7,  # Dự đoán 7 ngày
    'hidden_size': 50,
    'quantiles': [0.1, 0.5, 0.9],  # Các quantile cần dự đoán
    'dropout': 0.3,
    'layer_size': 2,
    'by_direction': False,
    'lr': 1e-3,
    'batch_size': 32,
    'num_epochs': 100,
    'context_size': 30,  # Sử dụng 30 ngày quá khứ
    'train_ratio': 0.8,  # Tỷ lệ train/validation
}

def load_and_preprocess_data():
    # Load data
    train = pd.read_csv('./data/rossmann-store-sales/train.csv', low_memory=False)
    store = pd.read_csv('./data/rossmann-store-sales/store.csv', low_memory=False)

    # Xử lý missing values
    store.CompetitionDistance = store.CompetitionDistance.fillna(store.CompetitionDistance.median())
    store.fillna(0, inplace=True)

    # Merge với store data
    train = pd.merge(train, store, on='Store')
    
    # Convert Date to datetime
    train['Date'] = pd.to_datetime(train['Date'])
    
    # Create time features
    train['Year'] = train['Date'].dt.year
    train['Month'] = train['Date'].dt.month
    train['Day'] = train['Date'].dt.day
    train['DayOfWeek'] = train['Date'].dt.dayofweek
    train['WeekOfYear'] = train['Date'].dt.isocalendar().week.astype(int)
    
    # Normalize numeric features
    numeric_features = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 
                       'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']
    
    scaler = StandardScaler()
    train[numeric_features] = scaler.fit_transform(train[numeric_features])
    
    # Normalize Sales per store
    train['Sales'] = train.groupby('Store')['Sales'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    return train

def create_mqrnn_dataset(df, target_col='Sales', covariate_cols=None):
    if covariate_cols is None:
        covariate_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear',
                         'CompetitionDistance', 'CompetitionOpenSinceMonth',
                         'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear',
                         'Promo', 'StateHoliday', 'SchoolHoliday', 'Open']
    
    df = df.sort_values(['Store', 'Date'])
    target_series = df.pivot(index='Date', columns='Store', values=target_col)
    covariate_df = df.pivot(index='Date', columns='Store', values=covariate_cols)
    covariate_df.columns = [f"{col}_{store}" for col, store in covariate_df.columns]
    return target_series, covariate_df

def split_train_val(df, train_ratio=0.8):
    # Sort by date to ensure chronological split
    df = df.sort_values('Date')
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    return train_df, val_df

def prepare_data_for_mqrnn(train_df, val_df, horizon_size=40, context_size=10):
    # Create datasets for train and validation
    train_target, train_covariates = create_mqrnn_dataset(train_df)
    val_target, val_covariates = create_mqrnn_dataset(val_df)
    
    # Create MQRNN datasets
    train_dataset = MQRNN_dataset(train_target, train_covariates, 
                                 horizon_size=horizon_size, 
                                 context_size=context_size)
    val_dataset = MQRNN_dataset(val_target, val_covariates,
                               horizon_size=horizon_size,
                               context_size=context_size)
    
    return train_dataset, val_dataset

def train_model(train_dataset, val_dataset, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    model = MQRNN(
        horizon_size=config['horizon_size'],
        hidden_size=config['hidden_size'],
        quantiles=config['quantiles'],
        columns=config['columns'],
        dropout=config['dropout'],
        layer_size=config['layer_size'],
        by_direction=config['by_direction']
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = model(batch)
                val_loss += loss.item()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config["num_epochs"]}], '
                  f'Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_mqrnn_model.pth')
    
    return model

def main():
    # Configuration
    config = {
        'horizon_size': 7,  # Dự đoán 7 ngày
        'hidden_size': 50,
        'quantiles': [0.1, 0.5, 0.9],  # Các quantile cần dự đoán
        'dropout': 0.3,
        'layer_size': 2,
        'by_direction': False,
        'lr': 1e-3,
        'batch_size': 32,
        'num_epochs': 100,
        'context_size': 30,  # Sử dụng 30 ngày quá khứ
        'train_ratio': 0.8,  # Tỷ lệ train/validation
    }
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Split into train and validation
    print("Splitting data into train and validation sets...")
    train_df, val_df = split_train_val(df, config['train_ratio'])
    
    # Prepare data for MQRNN
    print("Preparing data for MQRNN...")
    train_dataset, val_dataset = prepare_data_for_mqrnn(
        train_df, val_df,
        horizon_size=config['horizon_size'],
        context_size=config['context_size']
    )
    
    # Train model
    print("Training model...")
    model = train_model(train_dataset, val_dataset, config)
    print("Training completed!")

if __name__ == "__main__":
    main() 