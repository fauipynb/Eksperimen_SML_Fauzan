import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def prepare_data():
    # 1. Load Data
    raw_path = 'mobile_price_raw/mobile_price.csv'
    if not os.path.exists(raw_path):
        # Fallback for CI if data not yet downloaded
        url = "https://raw.githubusercontent.com/arpita-maji/Mobile-Price-Classification/master/mobile_price_range_data.csv"
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(raw_path)

    # 2. Split
    X = df.drop('price_range', axis=1)
    y = df['price_range']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Save
    output_dir = 'preprocessing/mobile_price_preprocessing'
    os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(f'{output_dir}/X_train.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

    joblib.dump(scaler, f'{output_dir}/scaler.joblib')
    print(f"Data preparation complete. Files saved to {output_dir}")

if __name__ == "__main__":
    prepare_data()
