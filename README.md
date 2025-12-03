# Transformer imputacija časovnih vrst

Kratek opis
-----------
Projekt vsebuje preprosto in reproducibilno cevovodje za imputacijo manjkajočih vrednosti v časovnih vrstah z uporabo prilagojenega Transformer modela. Namenjen je raziskavam in hitrim prototipom — priprava podatkov, učenje modela in izvajanje imputacije.

Kaj morate prenesti / namestiti
-------------------------------
- Python 3.9 ali novejši (priporočeno 3.9–3.11)
- GPU z nameščenim CUDA (opcijsko, priporočeno za hitrejše učenje)
- Namestite odvisnosti (primer `requirements.txt` spodaj)

Primarna Python knjižnica (priporočeno v `requirements.txt`):
```
torch
numpy
pandas
scikit-learn
joblib
matplotlib
pygrinder  # (če uporabljate MCAR masker iz repozitorija)
```

Hitri koraki za namestitev
-------------------------
```bash
git clone <repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate      # Windows (PowerShell)
pip install -r requirements.txt
```

Struktura repozitorija
----------------------
- `Transformer_model.py` — definicija TransformerImputer razreda (model, trening, validacija, impute metoda)
- `train.py` — skripta za pripravo podatkov, skaliranje, ustvarjanje drsnih oken in zagon treninga
- `support.py` — konfiguracije modelov in pomožne funkcije (npr. `create_sliding_windows`)
- `processed_data/` — izhod: train/val/test CSV-ji (per postaja)
- `scalers/` — shranjeni `StandardScaler` objekti (`scaler.pkl`)
- `models/` — shranjene uteži modela (`*.pth`)

Format vhodnih podatkov
-----------------------
- En CSV na postajo (npr. `station1.csv`) z vsaj:
  - stolpec `datetime` (parsable kot datum/čas)
  - enega ali več vrednostnih stolpcev (npr. `feature1`, `pm10`, ...)
- Primer (prvi stolpci):
```
datetime,feature1,feature2
2020-01-01 00:00:00,12.3,4.5
2020-01-01 01:00:00,11.8,4.2
...
```

Kako uporabiti (primeri)
------------------------

1. Priprava podatkov in učenje (primer):
```bash
python train.py   --data_dir path/data   --output_dir_models path/save/models   --output_dir_processed path/save/processed_data   --output_dir_scalers path/save/scalers   --stations station1 station2   --window_size 24   --step_size 1   --missing_rate 0.10
```
`train.py` v repoju naj bi naredil:
- branje CSV-jev iz `--data_dir`
- razdelitev na train/val/test
- fit `StandardScaler` na train set (shranjen v `--output_dir_scalers`)
- ustvarjanje drsnih oken s podanim `--window_size` in `--step_size`
- naključno maskiranje (če želite) validacijskega seta
- treniranje modelov in shranjevanje `*.pth` v `--output_dir_models`

2. Imputacija z že natreniranim modelom (kratka Python skripta)
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

Nasveti in opombe
-----------------
- Pri inferenci bodite pozorni na obliko podatkov (`(N, T, F)`).
- Shrani `scaler.pkl` vsake postaje — potreben je za povrnitev v originalno skalo.
- Če uporabljate GPU, preverite, da `torch.cuda.is_available()` in da naložite model z `map_location='cuda'` ali premaknete tenzorje na `device`.
- Če v modelu nastopijo NaN vrednosti pri imputaciji — preverite vhodne maske in da so vse NaN nadomeščene z 0 pred napovedjo (model pričakuje posebne maske).
- Prilagodite `batch_size`, `epochs` in `learning rate` glede na razpoložljivost strojne opreme.

Licenca
-------
Projekt lahko licencirate pod MIT licenco (ali drugo po izbiri).

Kontakt / prispevanje
---------------------
Če želite prispevati ali imate vprašanja, odprite `issue` ali pošljite pull request. Dodajte kratke, reproducibilne spremembe in testni primer.

