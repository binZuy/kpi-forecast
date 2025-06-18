import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from .Encoder import Encoder
from .Decoder import GlobalDecoder, LocalDecoder
from .data import MQRNN_Dataset
from .train_func import train_fn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class MQRNN(object):
    def __init__(self, 
                horizon_size:int, 
                hidden_size:int, 
                quantiles:list,
                columns:list, 
                dropout:float,
                layer_size:int,
                by_direction:bool,
                lr:float,
                batch_size:int, 
                num_epochs:int, 
                context_size:int, 
                covariate_size:int,
                device):
        print(f"device is: {device}")
        self.device = device
        self.horizon_size = horizon_size
        self.quantile_size = len(quantiles)
        self.quantiles = quantiles
        self.lr = lr 
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.covariate_size = covariate_size
        
        # Khởi tạo các components
        self.encoder = Encoder(
            horizon_size=horizon_size,
            covariate_size=covariate_size,
            hidden_size=hidden_size, 
            dropout=dropout,
            layer_size=layer_size,
            by_direction=by_direction,
            device=device
        )
        
        self.gdecoder = GlobalDecoder(
            hidden_size=hidden_size,
            covariate_size=covariate_size,
            horizon_size=horizon_size,
            context_size=context_size
        )
        
        self.ldecoder = LocalDecoder(
            covariate_size=covariate_size,
            quantile_size=self.quantile_size,
            context_size=context_size,
            quantiles=quantiles,
            horizon_size=horizon_size
        )
        
        # Chuyển model sang device và double precision
        self.encoder = self.encoder.to(device)
        self.gdecoder = self.gdecoder.to(device)
        self.ldecoder = self.ldecoder.to(device)
        
        self.encoder.double()
        self.gdecoder.double()
        self.ldecoder.double()
        
    def train(self, dataset: MQRNN_Dataset):
        
        train_fn(encoder=self.encoder, 
                gdecoder=self.gdecoder, 
                ldecoder=self.ldecoder,
                dataset=dataset,
                lr=self.lr,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                device=self.device)
        print("training finished")
    
    def predict(self, train_target_df, train_covariate_df, test_covariate_df, col_name):
        """
        Make predictions for a given column
        """
        input_target_tensor = torch.tensor(train_target_df[[col_name]].to_numpy())
        full_covariate = train_covariate_df.to_numpy()
        full_covariate_tensor = torch.tensor(full_covariate)

        next_covariate = test_covariate_df.to_numpy()
        next_covariate = next_covariate.reshape(-1, self.horizon_size * self.covariate_size)
        next_covariate_tensor = torch.tensor(next_covariate)

        # Move tensors to device
        input_target_tensor = input_target_tensor.to(self.device)
        full_covariate_tensor = full_covariate_tensor.to(self.device)
        next_covariate_tensor = next_covariate_tensor.to(self.device)

        with torch.no_grad():
            # Prepare input
            input_target_covariate_tensor = torch.cat([input_target_tensor, full_covariate_tensor], dim=1)
            input_target_covariate_tensor = input_target_covariate_tensor.unsqueeze(0)  # [1, seq_len, feature]
            # Nếu encoder dùng batch_first=True, không cần permute
            print(f"input_target_covariate_tensor shape: {input_target_covariate_tensor.shape}")
            outputs = self.encoder(input_target_covariate_tensor)  # [1, seq_len, hidden_size]
            enc_hs_flat = outputs.reshape(1, -1)  # [1, seq_len * hidden_size]
            next_covariate_flat = next_covariate_tensor.reshape(1, -1)  # [1, horizon_size * covariate_size]
            print(f"enc_hs_flat shape: {enc_hs_flat.shape}")
            print(f"next_covariate_flat shape: {next_covariate_flat.shape}")
            gdecoder_input = torch.cat([enc_hs_flat, next_covariate_flat], dim=1)  # [1, ...]
            gdecoder_output = self.gdecoder(gdecoder_input)
            print(f"gdecoder_output shape: {gdecoder_output.shape}")
            local_decoder_input = torch.cat([gdecoder_output, next_covariate_flat], dim=1)
            local_decoder_output = self.ldecoder(local_decoder_input)
            local_decoder_output = local_decoder_output.view(self.horizon_size, self.quantile_size)
            output_array = local_decoder_output.cpu().numpy()
            result_dict = {}
            for i in range(self.quantile_size):
                result_dict[self.quantiles[i]] = output_array[:,i]
            return result_dict