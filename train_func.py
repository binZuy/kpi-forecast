import torch
from .Encoder import Encoder
from .Decoder import GlobalDecoder, LocalDecoder
from .data import MQRNN_Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calc_loss(cur_series_covariate_tensor : torch.Tensor, 
            next_covariate_tensor: torch.Tensor,
            cur_real_vals_tensor: torch.Tensor, 
            encoder: Encoder,
            gdecoder: GlobalDecoder,
            ldecoder: LocalDecoder,
            device):
    loss = torch.tensor([0.0], device=device)

    cur_series_covariate_tensor = cur_series_covariate_tensor.double() #[batch_size, seq_len, 1+covariate_size]
    next_covariate_tensor = next_covariate_tensor.double() # [batch_size, seq_len, covariate_size * horizon_size]
    cur_real_vals_tensor = cur_real_vals_tensor.double() # [batch_size, seq_len, horizon_size]

    cur_series_covariate_tensor = cur_series_covariate_tensor.to(device)
    next_covariate_tensor = next_covariate_tensor.to(device)
    cur_real_vals_tensor = cur_real_vals_tensor.to(device)
    encoder.to(device)
    gdecoder.to(device)
    ldecoder.to(device)

    # cur_series_covariate_tensor = cur_series_covariate_tensor.permute(1,0,2) #[seq_len, batch_size, 1+covariate_size]
    # next_covariate_tensor = next_covariate_tensor.permute(1,0,2) #[seq_len, batch_size, covariate_size * horizon_size]
    # cur_real_vals_tensor = cur_real_vals_tensor.permute(1,0,2) #[seq_len, batch_size, horizon_size]
    enc_hs = encoder(cur_series_covariate_tensor) #[seq_len, batch_size, hidden_size]
    gdecoder_output = gdecoder(enc_hs, next_covariate_tensor)

    context_size = ldecoder.context_size
    
    quantile_size = ldecoder.quantile_size
    horizon_size = encoder.horizon_size
    total_loss = torch.tensor([0.0],device=device)

    local_decoder_input = torch.cat([gdecoder_output, next_covariate_tensor], dim=1) #[seq_len, batch_size,(horizon_size+1)*context_size + covariate_size * horizon_size]
    local_decoder_output = ldecoder( local_decoder_input) #[seq_len, batch_size, horizon_size* quantile_size]
    seq_len = local_decoder_output.shape[0]
    batch_size = local_decoder_output.shape[1]
    
    local_decoder_output = local_decoder_output.view(seq_len, batch_size, horizon_size, quantile_size) #[[seq_len, batch_size, horizon_size, quantile_size]]
    for i in range(quantile_size):
      p = ldecoder.quantiles[i]
      errors = cur_real_vals_tensor - local_decoder_output[:,:,:,i]
      cur_loss = torch.max( (p-1)*errors, p*errors ) # CAUTION
      total_loss += torch.sum(cur_loss)
    return total_loss


def train_fn(encoder, gdecoder, ldecoder, dataset, lr, batch_size, num_epochs, device):
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    gdecoder_optimizer = torch.optim.Adam(gdecoder.parameters(), lr=lr)
    ldecoder_optimizer = torch.optim.Adam(ldecoder.parameters(), lr=lr)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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

            encoder_input = encoder_input.to(device).double()
            future_covariate = future_covariate.to(device).double()
            target = target.to(device).double()

            encoder_optimizer.zero_grad()
            gdecoder_optimizer.zero_grad()
            ldecoder_optimizer.zero_grad()

            # Forward encoder
            enc_hs = encoder(encoder_input)  # [batch_size, context_size, hidden_size]

            # Lấy hidden cuối cùng của encoder
            enc_hs_last = enc_hs[:, -1:, :]  # [batch_size, 1, hidden_size]
            # Lặp lại hidden cho mỗi bước dự báo
            enc_hs_repeated = enc_hs_last.repeat(1, future_covariate.shape[1], 1)  # [batch_size, horizon_size, hidden_size]

            # Ghép với future_covariate
            # hidden_and_covariate = torch.cat([enc_hs_repeated, future_covariate], dim=2)  # [batch_size, horizon_size, hidden_size+num_features]

            # Global decoder
            gdecoder_output = gdecoder(enc_hs, future_covariate)

            # Local decoder (nếu cần)
            # local_decoder_input = torch.cat([gdecoder_output, future_covariate], dim=2)
            # local_decoder_output = ldecoder(local_decoder_input)

            # Tính loss (giả sử bạn có hàm loss phù hợp)
            loss = calc_loss(encoder_input, future_covariate, target, encoder, gdecoder, ldecoder, device)

            loss.backward()
            encoder_optimizer.step()
            gdecoder_optimizer.step()
            ldecoder_optimizer.step()

            epoch_loss_sum += loss.item()
            total_samples += encoder_input.shape[0]

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

# Ví dụ sử dụng trong notebook:
def evaluate_predictions(model, X_test, y_test):
    """
    Đánh giá model trên tập test
    
    Parameters:
    -----------
    model : object
        Model đã train
    X_test : array-like
        Dữ liệu test
    y_test : array-like
        Giá trị thực tế của test set
    """
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Tính metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # In kết quả
    print("Kết quả đánh giá model:")
    print("-" * 30)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Thực tế', marker='o')
    plt.plot(y_pred, label='Dự đoán', marker='x')
    plt.title('So sánh giá trị thực tế và dự đoán')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return metrics