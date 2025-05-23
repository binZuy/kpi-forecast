import torch
from data import load_processed_data, MQRNN_Dataset
from MQRNN import MQRNN

def predict(npz_path, model_path, horizon_size, covariate_size, quantiles, context_size, device):
    # Load data
    X, y, covariates = load_processed_data(npz_path)
    dataset = MQRNN_Dataset(X, y, covariates)
    # Lấy 1 sample để demo
    encoder_input, future_covariate, _ = dataset[0]
    encoder_input = encoder_input.unsqueeze(1).to(device)  # [input_window, 1, 1+num_covariates]
    future_covariate = future_covariate.unsqueeze(0).to(device)  # [1, output_window, num_covariates]

    # Load model
    model = MQRNN(
        horizon_size=horizon_size,
        hidden_size=64,
        quantiles=quantiles,
        columns=None,
        dropout=0.1,
        layer_size=2,
        by_direction=False,
        lr=1e-3,
        batch_size=1,
        num_epochs=1,
        context_size=context_size,
        covariate_size=covariate_size,
        device=device
    )
    model.encoder.load_state_dict(torch.load(model_path, map_location=device))
    model.encoder.eval()
    model.gdecoder.eval()
    model.ldecoder.eval()

    with torch.no_grad():
        outputs = model.encoder(encoder_input.permute(1,0,2))  # [seq_len, 1, hidden_size]
        hidden = outputs[-1].unsqueeze(0)  # [1, 1, hidden_size]
        next_covariate = future_covariate.reshape(1, -1)
        gdecoder_input = torch.cat([hidden.reshape(1, -1), next_covariate], dim=1)
        gdecoder_output = model.gdecoder(gdecoder_input)
        local_decoder_input = torch.cat([gdecoder_output, next_covariate], dim=1)
        local_decoder_output = model.ldecoder(local_decoder_input)
        local_decoder_output = local_decoder_output.view(horizon_size, len(quantiles))
        print("Predicted quantiles for horizon:", local_decoder_output.cpu().numpy())
        return local_decoder_output.cpu().numpy()
