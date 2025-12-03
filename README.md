# Transformer imputacija časovnih vrst

## Kratek opis
Projekt vsebuje Transformer model z rotacijskim pozicijskim kodiranjem (RoPE), namenjen zapolnjevanju manjkajočih vrednosti v časovnih vrstah. Model uporablja self-attention mehanizem ter Mixed Imputation Training (MIT) strategijo, ki omogoča stabilno in učinkovito učenje.

---

## Za delovanje potrebujemo

- Python 3.8 ali novejši  
- GPU z nameščenim CUDA (priporočeno)  
- Odvisnosti navedene v `requirements.txt`

### Namestitev
```
pip install -r requirements.txt
```

---

## Struktura projekta
```
.
├── Transformer_model.py    # Implementacija Transformer modela
├── train.py                # Skripta za učenje
├── support.py              # Pomožne funkcije (create_sliding_windows, itd.)
├── requirements.txt        # Python odvisnosti
└── README.md               # Ta dokument
```

---

## Format vhodnih podatkov

- En CSV na postajo (npr. `station1.csv`) z vsaj:
  - stolpec `datetime`
  - enega ali več numeričnih stolpcev (`feature1`, `pm10`, ...)

### Primer
```
datetime,feature1,feature2
2020-01-01 00:00:00,12.3,4.5
2020-01-01 01:00:00,11.8,4.2
...
```

---

## Kako uporabiti

### 1. Priprava podatkov in učenje

**POMEMBNO:** Pred zagonom v datoteki `train.py` uredite:

```python
data_dir = 'path/data'
output_dir_models = 'path/save/models'
output_dir_processed = 'path/save/processed_data'
output_dir_scalers = 'path/save/scalers'

stations = ['station1', 'station2', 'station3']
features = ['feature1']

window_size = 24
step_size = 1
missing_rate = 0.10
```

Nato zaženite:

```
python train.py
```

---

### 2. Imputacija z že natreniranim modelom

```python
import joblib
import numpy as np
import pandas as pd
import torch
from Transformer_model import TransformerImputer, TransformerConfig
from support import create_sliding_windows

# Naloži scaler in model
scaler = joblib.load("path/save/scalers/station1/scaler.pkl")

# Konfiguracija (mora biti enaka kot pri učenju)
config = TransformerConfig(
    block_size=24,
    n_features=1,
    d_model=128,
    n_layer=3,
    n_head=4,
    dropout=0.1
)

model = TransformerImputer(config)
model.load_state_dict(torch.load(
    "path/save/models/transformer/station1/your_model.pth",
    map_location="cpu"
))

# Priprava podatkov
df = pd.read_csv("test_data.csv")
data = df[['feature1']].values

# Normalizacija
data_scaled = scaler.transform(data)

# Drsna okna
windows = create_sliding_windows(
    data_scaled,
    window_size=24,
    step_size=1,
    n_features=data.shape[1]
)

dataset = {"X": windows}
imputed = model.impute(dataset)

# Oblika rezultata
imputed_unscaled = scaler.inverse_transform(
    imputed.reshape(-1, imputed.shape[-1])
).reshape(imputed.shape)

print("Imputirani podatki:", imputed_unscaled.shape)
```

---

## Konfiguracija modela

### Parametri `TransformerConfig`

| Parameter     | Privzeto | Opis |
|---------------|----------|------|
| block_size    | 24       | Dolžina okna |
| n_features    | 10       | Število značilk |
| d_model       | 128      | Dimenzija vgraditve |
| n_layer       | 3        | Število Transformer blokov |
| n_head        | 4        | Število attention glav |
| dropout       | 0.1      | Stopnja dropouta |
| bias          | True     | Bias v linearnih plasteh |

### Parametri metode `fit()`

| Parameter     | Privzeto | Opis |
|---------------|----------|------|
| epochs        | 300      | Število epoh |
| batch_size    | 128      | Velikost batch-a |
| initial_lr    | 1e-3     | Začetni LR |
| patience      | 250      | Early stopping |
| min_delta     | 0.0      | Minimalno izboljšanje |

---

## Strategija učenja

Model uporablja **Mixed Imputation Training (MIT)**:

- naključno se zakrije 25% opaženih vrednosti  
- MIT loss: napaka na umetno zakritih vrednostih  
- ORT loss: rekonstrukcija opaženih vrednosti  

Validacija poteka na umetno zamaskiranih validacijskih podatkih.

---

## Optimizacije

Model vključuje več optimizacij:

- `torch.compile()` za hitrejše izvajanje
- mixed precision training (AMP)
- predpomnjenje diagonalne attention maske
- TF32 in cuDNN optimizacije
- gradient clipping

---

## Komponente modela

### `RotaryPositionalEncoding`
Rotacijsko pozicijsko kodiranje.

### `SelfAttention`
Multi-head attention z:
- RoPE kodiranjem
- diagonal masking  
- flash attention optimizacijo

### `TransformerImputer`
Vključuje metode:
- `fit()` – učenje  
- `impute()` – imputacija  
- `number_of_params()` – štetje parametrov  



