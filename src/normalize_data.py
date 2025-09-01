import os
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def parse_args():
    p = argparse.ArgumentParser(description="Normalise X_train et X_test puis sauvegarde des versions _scaled.")
    p.add_argument("--in-train", default="data/processed_data/X_train.csv")
    p.add_argument("--in-test",  default="data/processed_data/X_test.csv")
    p.add_argument("--out-dir",  default="data/processed")
    p.add_argument("--save-scaler", action="store_true")
    return p.parse_args()

def main(in_train: str, in_test: str, out_dir: str, save_scaler: bool):
    # Chargement
    X_train = pd.read_csv(in_train)
    X_test = pd.read_csv(in_test)

    # Validation basique
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Les colonnes de X_train et X_test ne correspondent pas.")

    # Fit sur train, transform sur train + test
    scaler = StandardScaler()
    X_train_scaled_arr = scaler.fit_transform(X_train.values)
    X_test_scaled_arr  = scaler.transform(X_test.values)

    # Reconstruction DataFrames avec les mêmes colonnes
    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=X_train.columns, index=X_train.index)
    X_test_scaled  = pd.DataFrame(X_test_scaled_arr,  columns=X_test.columns,  index=X_test.index)

    # Sauvegarde
    os.makedirs(out_dir, exist_ok=True)
    X_train_scaled.to_csv(os.path.join(out_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(out_dir, "X_test_scaled.csv"), index=False)

    # Sauvegarder le scaler pour réutilisation en prod
    if save_scaler:
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.joblib")

    print(f"Sauvegardé: {os.path.join(out_dir, 'X_train_scaled.csv')} et X_test_scaled.csv")
    if save_scaler:
        print("Scaler sauvegardé dans models/scaler.joblib")

if __name__ == "__main__":
    args = parse_args()
    main(args.in_train, args.in_test, args.out_dir, args.save_scaler)
