# Mobile Price Classification - Eksperimen SML

Repository ini berisi eksperimen Machine Learning untuk klasifikasi rentang harga ponsel berdasarkan spesifikasi teknis.

## Struktur Repository

- `mobile_price_raw/`: Berisi dataset mentah (`mobile_price.csv`).
- `preprocessing/`:
    - `Eksperimen_Fauzan.ipynb`: Notebook eksperimen lengkap (EDA, Preprocessing, Modeling).
    - `automate_Fauzan.py`: Script Python untuk otomasi data pipeline.
    - `mobile_price_preprocessing/`: **Dataset hasil preprocessing** (X_train, X_test, y_train, y_test, scaler.joblib) - *Wajib untuk submission Dicoding MSML*.

## Cara Menjalankan Preprocessing
Untuk menghasilkan kembali dataset hasil preprocessing, jalankan perintah berikut di root directory:
```bash
python preprocessing/automate_Fauzan.py
```
