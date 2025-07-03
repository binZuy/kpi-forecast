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
        # Quantile loss: max(q*errors, (q-1)*errors)
        cur_loss = torch.max(q * errors, (q - 1) * errors)
        total_loss += torch.mean(cur_loss)  # Sử dụng mean thay vì sum
    
    # Thêm debug để kiểm tra
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("WARNING: Loss is NaN or Inf!")
        print("local_decoder_output min/max:", local_decoder_output.min().item(), local_decoder_output.max().item())
        print("target min/max:", target.min().item(), target.max().item())
        print("errors min/max:", errors.min().item(), errors.max().item())
    
    return total_loss


def train_fn(encoder, gdecoder, ldecoder, dataset, lr, batch_size, num_epochs, device):
    # Giảm learning rate để tránh gradient explosion
    lr = min(lr, 0.0001)  # Giới hạn learning rate tối đa thấp hơn
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    gdecoder_optimizer = torch.optim.Adam(gdecoder.parameters(), lr=lr, weight_decay=1e-4)
    ldecoder_optimizer = torch.optim.Adam(ldecoder.parameters(), lr=lr, weight_decay=1e-4)

    # Thêm learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        encoder_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    for epoch in range(num_epochs):
        encoder.train()
        gdecoder.train()
        ldecoder.train()
        epoch_loss_sum = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(data_loader):
            # unpack batch
            encoder_input, future_covariate, target = batch
            # encoder_input: [batch_size, context_size, 1+num_features]
            # future_covariate: [batch_size, horizon_size, num_features]
            # target: [batch_size, horizon_size]
            
            encoder_input = encoder_input.to(device)
            future_covariate = future_covariate.to(device)
            target = target.to(device)
            
            # Kiểm tra dữ liệu đầu vào
            if torch.isnan(encoder_input).any() or torch.isinf(encoder_input).any():
                print(f"WARNING: encoder_input contains NaN/Inf at epoch {epoch}, batch {batch_idx}")
                continue
                
            if torch.isnan(future_covariate).any() or torch.isinf(future_covariate).any():
                print(f"WARNING: future_covariate contains NaN/Inf at epoch {epoch}, batch {batch_idx}")
                continue
                
            if torch.isnan(target).any() or torch.isinf(target).any():
                print(f"WARNING: target contains NaN/Inf at epoch {epoch}, batch {batch_idx}")
                continue
            
            encoder_optimizer.zero_grad()
            gdecoder_optimizer.zero_grad()
            ldecoder_optimizer.zero_grad()
            
            # Forward encoder
            enc_hs = encoder(encoder_input)  # [batch_size, context_size, hidden_size]

            batch_size = enc_hs.shape[0]
            enc_hs_flat = enc_hs.reshape(batch_size, -1)  # [batch_size, context_size * hidden_size]
            future_covariate_flat = future_covariate.reshape(batch_size, -1)  # [batch_size, horizon_size * covariate_size]

            # Debug: Kiểm tra input của GlobalDecoder
            if batch_idx == 0 and epoch % 5 == 0:
                print(f"Epoch {epoch}, Batch 0:")
                print(f"  enc_hs_flat shape: {enc_hs_flat.shape}")
                print(f"  future_covariate_flat shape: {future_covariate_flat.shape}")
                print(f"  enc_hs_flat min/max: {enc_hs_flat.min().item():.4f}/{enc_hs_flat.max().item():.4f}")
                print(f"  future_covariate_flat min/max: {future_covariate_flat.min().item():.4f}/{future_covariate_flat.max().item():.4f}")
                
                # Kiểm tra xem có giá trị quá lớn không
                if future_covariate_flat.max().item() > 10:
                    print(f"  WARNING: future_covariate_flat có giá trị lớn: {future_covariate_flat.max().item():.4f}")
                    print(f"  future_covariate_flat std: {future_covariate_flat.std().item():.4f}")

            # Concat để tạo input cho GlobalDecoder
            gdecoder_input = torch.cat([enc_hs_flat, future_covariate_flat], dim=1)  # [batch_size, ...]
            
            # Debug: Kiểm tra input của GlobalDecoder
            if batch_idx == 0 and epoch % 5 == 0:
                print(f"  gdecoder_input shape: {gdecoder_input.shape}")
                print(f"  gdecoder_input min/max: {gdecoder_input.min().item():.4f}/{gdecoder_input.max().item():.4f}")
            
            gdecoder_output = gdecoder(gdecoder_input)  # [batch_size, ...]
            
            # Debug: Kiểm tra output của GlobalDecoder
            if batch_idx == 0 and epoch % 5 == 0:
                print(f"  gdecoder_output shape: {gdecoder_output.shape}")
                print(f"  gdecoder_output min/max: {gdecoder_output.min().item():.4f}/{gdecoder_output.max().item():.4f}")
            
            # Flatten future_covariate lại nếu cần cho local decoder
            local_decoder_input = torch.cat([gdecoder_output, future_covariate_flat], dim=1)
            
            # Debug: Kiểm tra input của LocalDecoder
            if batch_idx == 0 and epoch % 5 == 0:
                print(f"  local_decoder_input shape: {local_decoder_input.shape}")
                print(f"  local_decoder_input min/max: {local_decoder_input.min().item():.4f}/{local_decoder_input.max().item():.4f}")
            
            local_decoder_output = ldecoder(local_decoder_input)
            
            # Debug: Kiểm tra output của LocalDecoder
            if batch_idx == 0 and epoch % 5 == 0:
                print(f"  local_decoder_output shape: {local_decoder_output.shape}")
                print(f"  local_decoder_output min/max: {local_decoder_output.min().item():.4f}/{local_decoder_output.max().item():.4f}")
                print(f"  Target min/max: {target.min().item():.4f}/{target.max().item():.4f}")

            # Tính loss
            loss = calc_loss(local_decoder_output, target, ldecoder.quantiles)
            
            # Kiểm tra loss có hợp lệ không
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e3:
                print(f"ERROR: Loss is {loss.item()} at epoch {epoch}, batch {batch_idx}")
                print(f"  local_decoder_output min/max: {local_decoder_output.min().item():.4f}/{local_decoder_output.max().item():.4f}")
                print(f"  target min/max: {target.min().item():.4f}/{target.max().item():.4f}")
                continue  # Bỏ qua batch này
            
            # Debug: In loss sau khi đã tính toán
            if batch_idx == 0 and epoch % 5 == 0:
                print(f"  Loss: {loss.item():.4f}")
                print("  " + "="*50)

            loss.backward()
            
            # Clip gradients mạnh hơn để tránh gradient explosion
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.001)
            torch.nn.utils.clip_grad_norm_(gdecoder.parameters(), max_norm=0.001)
            torch.nn.utils.clip_grad_norm_(ldecoder.parameters(), max_norm=0.001)
            
            # Kiểm tra gradients
            if batch_idx == 0 and epoch % 5 == 0:
                total_grad_norm = 0
                for name, param in encoder.named_parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item()
                print(f"  Total encoder grad norm: {total_grad_norm:.6f}")
                
                # Kiểm tra gradient explosion
                if total_grad_norm > 1:
                    print(f"  WARNING: Gradient norm too large: {total_grad_norm:.6f}")

            encoder_optimizer.step()
            gdecoder_optimizer.step()
            ldecoder_optimizer.step()
            
            epoch_loss_sum += loss.item()
            total_samples += encoder_input.shape[0]
        
        if (epoch+1)%5 == 0:
            avg_loss = epoch_loss_sum/total_samples
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # Cập nhật learning rate dựa trên loss
            scheduler.step(avg_loss)
        
            # Kiểm tra learning rate
            print(f"  Learning rate: {encoder_optimizer.param_groups[0]['lr']:.6f}")

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
