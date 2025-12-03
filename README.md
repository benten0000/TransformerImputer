# Transformer imputacija časovnih vrst

Kratek opis
-----------
Projekt vsebuje model in učno zanko namenjeno zapolnjevanju manjkajočih vrednosti v časovnih vrstah z uporabo Transformer modela.

Za delovanje potrebujemo
-------------------------------
- Python 3.9 ali novejši
- GPU z nameščenim CUDA
- Odvisnosti so navedene v `requirements.txt`

Format vhodnih podatkov
-----------------------
- En CSV na postajo (npr. `station1.csv`) z vsaj:
  - stolpec `datetime` (parsable kot datum/čas)
  - enega ali več vrednostnih stolpcev (npr. `feature1`, `pm10`, ...)
- Primer:
```
datetime,feature1,feature2
2020-01-01 00:00:00,12.3,4.5
2020-01-01 01:00:00,11.8,4.2
...
```

Kako uporabiti
------------------------

1. Priprava podatkov in učenje (primer):
```bash
python train.py   --data_dir path/data   --output_dir_models path/save/models   --output_dir_processed path/save/processed_data   --output_dir_scalers path/save/scalers   --stations station1 station2   --window_size 24   --step_size 1   --missing_rate 0.10
```

2. Imputacija z že natreniranim modelom
```python
import joblib
import numpy as np
import torch
from Transformer_model import TransformerImputer
from support import create_sliding_windows

# naloži scaler in model
scaler = joblib.load("path/save/scalers/station1/scaler.pkl")
config = ...  # ista konfiguracija kot pri treningu
model = TransformerImputer(config)
model.load_state_dict(torch.load("path/save/models/transformer/station1/your_model.pth", map_location="cpu"))

# pripravi podatke (na primer iz CSV)
# predpostavimo, da imamo N x F polje 'data' (nenormalizirano)
data_scaled = scaler.transform(data)
windows = create_sliding_windows(data_scaled, window_size=24, step=1, n_features=data.shape[1])

# vstavi NaN v windows tam, kjer želite imputirati, nato:
dataset = {"X": windows}
imputed = model.impute(dataset)  # vrne numpy array v skalirani skali

# obratna transformacija
# reshape, če je potrebno, nato inverse_transform po feature-osi
# (odvisno od oblike heads — običajno je output v obliki (N, T, F))
imputed_unscaled = scaler.inverse_transform(imputed.reshape(-1, imputed.shape[-1])).reshape(imputed.shape)
```



