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
        super(MQRNN, self).__init__()
        
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
        
        # Chuyển model sang double precision
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
    
    def _calc_loss(self, predictions, target):
        """
        Tính quantile loss
        """
        total_loss = torch.tensor([0.0], device=self.device)
        for i in range(self.quantile_size):
            p = self.quantiles[i]
            errors = target - predictions[:,:,:,i]
            cur_loss = torch.max((p-1)*errors, p*errors)
            total_loss += torch.sum(cur_loss)
            
        return total_loss
    
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
            input_target_covariate_tensor = torch.unsqueeze(input_target_covariate_tensor, dim=0)
            input_target_covariate_tensor = input_target_covariate_tensor.permute(1,0,2)
            
            # Get encoder output
            outputs = self.encoder(input_target_covariate_tensor)
            hidden = torch.unsqueeze(outputs[-1], dim=0)
            
            # Prepare decoder input
            next_covariate_tensor = torch.unsqueeze(next_covariate_tensor, dim=0)
            gdecoder_input = torch.cat([hidden, next_covariate_tensor], dim=2)
            
            # Get predictions
            gdecoder_output = self.gdecoder(gdecoder_input)
            local_decoder_input = torch.cat([gdecoder_output, next_covariate_tensor], dim=2)
            local_decoder_output = self.ldecoder(local_decoder_input)
            
            # Reshape output
            local_decoder_output = local_decoder_output.view(self.horizon_size, self.quantile_size)
            output_array = local_decoder_output.cpu().numpy()
            
            # Create result dictionary
            result_dict = {}
            for i in range(self.quantile_size):
                result_dict[self.quantiles[i]] = output_array[:,i]
                
            return result_dict