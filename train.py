import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from Transformer_model import TransformerImputer, TransformerConfig
from pypots.imputation import SAITS
import joblib
from support import transformer_configs, saits_configs, create_sliding_windows, knn_configs

# poti
data_dir = 
output_dir1 = 
output_dir2 = 
output_dir3 = 
stations = ['E410', 'E411', 'E412', 'E415', 'E802']

models = ['transformer']
features = ['PM2.5']

for selected_station in stations:
    print(f"Obdelava postaje: {selected_station}")
    
    file_path = os.path.join(data_dir, f'{selected_station}.csv')
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    data = df[features].values
    datetime = df['datetime'].values

    train_size = int(0.80 * len(data))
    val_size = int(0.10 * len(data))
    
    data_train_original = data[:train_size]
    data_val_original = data[train_size:train_size + val_size]
    data_test_original = data[train_size + val_size:]
    
    datetime_train = datetime[:train_size]
    datetime_val = datetime[train_size:train_size + val_size]
    datetime_test = datetime[train_size + val_size:]

    data_output_dir = os.path.join(output_dir2, selected_station)
    os.makedirs(data_output_dir, exist_ok=True)

    train_df = pd.DataFrame(data_train_original, columns=features)
    train_df['datetime'] = datetime_train
    train_df.to_csv(os.path.join(data_output_dir, 'train_data.csv'), index=False)

    val_df = pd.DataFrame(data_val_original, columns=features)
    val_df['datetime'] = datetime_val
    val_df.to_csv(os.path.join(data_output_dir, 'val_data.csv'), index=False)

    test_df = pd.DataFrame(data_test_original, columns=features)
    test_df['datetime'] = datetime_test
    test_df.to_csv(os.path.join(data_output_dir, 'test_data.csv'), index=False)

    scaler = StandardScaler()
    data_train_scaled = scaler.fit_transform(data_train_original)
    data_val_scaled = scaler.transform(data_val_original)
    data_test_scaled = scaler.transform(data_test_original)
    
    scaler_output_dir = os.path.join(output_dir3, selected_station)
    os.makedirs(scaler_output_dir, exist_ok=True)
    
    scaler_path = os.path.join(scaler_output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  Shranjen scaler za postajo {selected_station}")

    window_size = 24
    step_size = 1
    n_features = len(features)


    def create_sliding_windows(data, window_size, step_size, n_features):
        windows = []
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data[i:i + window_size]
            windows.append(window.reshape(1, window_size, n_features))
        return np.concatenate(windows, axis=0)
    
    X_train = create_sliding_windows(data_train_scaled, window_size, step_size, n_features)
    X_val_ori = create_sliding_windows(data_val_scaled, window_size, step_size, n_features)
    X_test_ori = create_sliding_windows(data_test_scaled, window_size, step_size, n_features)
    
    missing_rate = 0.10
    X_val_masked = X_val_ori.copy()
    for i in range(X_val_masked.shape[0]):
        mask = np.random.random((window_size, n_features)) < missing_rate
        X_val_masked[i][mask] = np.nan
    
    dataset_train = {
        "X": X_train
    }
    dataset_val = {
        "X": X_val_masked,
        "X_ori": X_val_ori
    }

    for model_type in models:
        print(f"  Učenje {model_type} modelov")
        model_output_dir = os.path.join(output_dir1, model_type, selected_station)
        os.makedirs(model_output_dir, exist_ok=True)
        if model_type == "transformer":
            for config in transformer_configs:
                print(f"Učenje {config['name']}")
                dataset_train_transformer = {"X": X_train}
                dataset_val_transformer = {"X": X_val_masked}
                model = TransformerImputer(config["config"])
                print("stevilo parametrov: ", model.number_of_params())
                model.fit(dataset_train, validation_data=dataset_val, **config["fit_params"])
                model_path = os.path.join(model_output_dir, f'{config["name"]}.pth')
                torch.save(model.state_dict(), model_path)
    print(f"Postaja {selected_station} končana")
print("Učenje vseh modelov končano")
