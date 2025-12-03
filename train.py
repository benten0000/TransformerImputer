import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from Transformer_model import TransformerImputer
import joblib
from support import transformer_configs, create_sliding_windows

data_dir = 'path/data'
output_dir_models = 'path/save/models'
output_dir_processed = 'path/save/processed_data'
output_dir_scalers = 'path/save/scalers'

stations = ['station1', 'station2', 'station3']
features = ['feature1']
window_size = 24
step_size = 1
missing_rate = 0.10
train_ratio, val_ratio = 0.80, 0.10


for station in stations:
    df = pd.read_csv(os.path.join(data_dir, f'{station}.csv'), parse_dates=['datetime'])
    data, datetime_values = df[features].values, df['datetime'].values
    
    train_size = int(train_ratio * len(data))
    val_size = int(val_ratio * len(data))
    
    data_splits = {
        'train': (data[:train_size], datetime_values[:train_size]),
        'val': (data[train_size:train_size + val_size], datetime_values[train_size:train_size + val_size]),
        'test': (data[train_size + val_size:], datetime_values[train_size + val_size:])
    }
    
    
    data_output_dir = os.path.join(output_dir_processed, station)
    os.makedirs(data_output_dir, exist_ok=True)
    for split_name, (split_data, split_datetime) in data_splits.items():
        pd.DataFrame(split_data, columns=features).assign(datetime=split_datetime).to_csv(
            os.path.join(data_output_dir, f'{split_name}_data.csv'), index=False)
    
    
    scaler = StandardScaler()
    data_train_scaled = scaler.fit_transform(data_splits['train'][0])
    data_val_scaled = scaler.transform(data_splits['val'][0])
    
    scaler_dir = os.path.join(output_dir_scalers, station)
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, 'scaler.pkl'))
    
    
    X_train = create_sliding_windows(data_train_scaled, window_size, step_size, len(features))
    X_val_ori = create_sliding_windows(data_val_scaled, window_size, step_size, len(features))
    
    
    X_val_masked = X_val_ori.copy()
    for i in range(X_val_masked.shape[0]):
        mask = np.random.random((window_size, len(features))) < missing_rate
        X_val_masked[i][mask] = np.nan
    
    dataset_train = {"X": X_train}
    dataset_val = {"X": X_val_masked, "X_ori": X_val_ori}
    
    
    model_output_dir = os.path.join(output_dir_models, 'transformer', station)
    os.makedirs(model_output_dir, exist_ok=True)
    
    for config in transformer_configs:
        print(f"  Training {config['name']} - params: {(model := TransformerImputer(config['config'])).number_of_params()}")
        model.fit(dataset_train, validation_data=dataset_val, **config["fit_params"])
        torch.save(model.state_dict(), os.path.join(model_output_dir, f'{config["name"]}.pth'))
    


