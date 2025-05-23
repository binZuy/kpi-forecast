import numpy as np
import torch
from torch.utils.data import Dataset

def load_processed_data(npz_path):
    """
    Load preprocessed data from .npz file
    """
    data = np.load(npz_path)
    X = data['X']  # [num_samples, input_window]
    y = data['y']  # [num_samples, output_window]
    covariates = data['covariates']  # [num_samples, input_window+output_window, num_covariates]
    return X, y, covariates

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


