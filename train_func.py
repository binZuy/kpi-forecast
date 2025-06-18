import torch
from .Encoder import Encoder
from .Decoder import GlobalDecoder, LocalDecoder
from .data import MQRNN_Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calc_loss(local_decoder_output, target, quantiles):
    # local_decoder_output: [batch_size, horizon_size * quantile_size]
    # target: [batch_size, horizon_size]
    batch_size, output_dim = local_decoder_output.shape
    horizon_size = target.shape[1]
    quantile_size = len(quantiles)
    # Reshape lại nếu cần
    local_decoder_output = local_decoder_output.view(batch_size, horizon_size, quantile_size)
    total_loss = 0.0
    for i, q in enumerate(quantiles):
        errors = target - local_decoder_output[:, :, i]
        cur_loss = torch.max((q-1)*errors, q*errors)
        total_loss += torch.sum(cur_loss)
    return total_loss


def train_fn(encoder, gdecoder, ldecoder, dataset, lr, batch_size, num_epochs, device):
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    gdecoder_optimizer = torch.optim.Adam(gdecoder.parameters(), lr=lr)
    ldecoder_optimizer = torch.optim.Adam(ldecoder.parameters(), lr=lr)

    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    for epoch in range(num_epochs):
        encoder.train()
        gdecoder.train()
        ldecoder.train()
        epoch_loss_sum = 0.0
        total_samples = 0

        for batch in data_loader:
            # unpack batch
            encoder_input, future_covariate, target = batch
            # encoder_input: [batch_size, context_size, 1+num_features]
            # future_covariate: [batch_size, horizon_size, num_features]
            # target: [batch_size, horizon_size]

            encoder_input = encoder_input.to(device)
            future_covariate = future_covariate.to(device)
            target = target.to(device)

            encoder_optimizer.zero_grad()
            gdecoder_optimizer.zero_grad()
            ldecoder_optimizer.zero_grad()

            # Forward encoder
            enc_hs = encoder(encoder_input)  # [batch_size, context_size, hidden_size]

            batch_size = enc_hs.shape[0]
            enc_hs_flat = enc_hs.reshape(batch_size, -1)  # [batch_size, context_size * hidden_size]
            future_covariate_flat = future_covariate.reshape(batch_size, -1)  # [batch_size, horizon_size * covariate_size]

            # Concat để tạo input cho GlobalDecoder
            gdecoder_input = torch.cat([enc_hs_flat, future_covariate_flat], dim=1)  # [batch_size, ...]
            gdecoder_output = gdecoder(gdecoder_input)  # [batch_size, ...]
            
            # Flatten future_covariate lại nếu cần cho local decoder
            local_decoder_input = torch.cat([gdecoder_output, future_covariate_flat], dim=1)
            local_decoder_output = ldecoder(local_decoder_input)

            # Tính loss
            loss = calc_loss(local_decoder_output, target, ldecoder.quantiles)

            loss.backward()
            encoder_optimizer.step()
            gdecoder_optimizer.step()
            ldecoder_optimizer.step()

            epoch_loss_sum += loss.item()
            total_samples += encoder_input.shape[0]
        if (epoch+1)%5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss_sum/total_samples:.4f}")

def calculate_metrics(y_true, y_pred):
    """
    Tính toán các metrics đánh giá cho bài toán forecasting
    
    Parameters:
    -----------
    y_true : array-like
        Giá trị thực tế
    y_pred : array-like
        Giá trị dự đoán
        
    Returns:
    --------
    dict
        Dictionary chứa các metrics
    """
    # Đảm bảo input là numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Tính các metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Tính MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
