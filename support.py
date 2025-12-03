from Transformer_model import TransformerConfig
import numpy as np

transformer_configs = [
    {
        "name": "transformer_v1",
        "config": TransformerConfig(
            block_size=24,
            n_features=1,
            d_model=128,
            n_layer=3,
            n_head=8,
            dropout=0.10,
            bias=True
        ),
        "fit_params": {
            "epochs": 200,
            "batch_size": 256,
            "initial_lr": 2e-4
        }
    },
    {
        "name": "transformer_v2",
        "config": TransformerConfig(
            block_size=24,
            n_features=1,
            d_model=256,
            n_layer=3,
            n_head=8,
            dropout=0.10,
            bias=True
        ),
        "fit_params": {
            "epochs": 200,
            "batch_size": 256,
            "initial_lr": 1.5e-4
        }
    },
    {
        "name": "transformer_v3",
        "config": TransformerConfig(
            block_size=24,
            n_features=1,
            d_model=512,
            n_layer=3,
            n_head=8,
            dropout=0.10,
            bias=True
        ),
        "fit_params": {
            "epochs": 200,
            "batch_size": 256,
            "initial_lr": 1e-4
        }
    }
]

def create_sliding_windows(data, window_size, step_size, n_features):
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i + window_size]
        windows.append(window.reshape(1, window_size, n_features))
    return np.concatenate(windows, axis=0)

def reconstruct_from_windows(imputed_windows, data_length, window_size, step_size, n_features):
        reconstructed = np.zeros((data_length, n_features))
        count = np.zeros(data_length)
        current_idx = 0
        for i in range(0, data_length - window_size + 1, step_size):
            if current_idx < len(imputed_windows):
                window = imputed_windows[current_idx]
                reconstructed[i:i + window_size] += window
                count[i:i + window_size] += 1
                current_idx += 1
        reconstructed = reconstructed / np.maximum(count[:, None], 1)
        return reconstructed
