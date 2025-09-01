import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def main(csv_path="/data/raw_data/raw.csv", out_dir="/data/processed_data", test_size=0.25, random_state=42):
    # Colonnes à garder
    cols = [
        "ave_flot_air_flow",
        "ave_flot_level",
        "iron_feed",
        "starch_flow",
        "amina_flow",
        "ore_pulp_flow",
        "ore_pulp_pH",
        "ore_pulp_density",
        "silica_concentrate",  # cible
    ]
    target = "silica_concentrate"

    # Charger le CSV + ignorer la colonne date ici
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # Vérifier que toutes les colonnes existent
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path}: {missing}")

    # Sélection
    df_sel = df[cols].copy()

    # Définir X et y
    X = df_sel.drop(columns=[target]).values
    y = df_sel[target].values

    # Split avec random_state fixe pour la reproductibilité
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Sauvegarde
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(X_train, columns=[c for c in cols if c != target]).to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test, columns=[c for c in cols if c != target]).to_csv(os.path.join(out_dir, "X_test.csv"), index=False)
    pd.Series(y_train, name=target).to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    pd.Series(y_test, name=target).to_csv(os.path.join(out_dir, "y_test.csv"), index=False)

    # Les shapes
    print(f"Saved CSVs to {out_dir}")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

if __name__ == "__main__":
    main(csv_path="data/raw_data/raw.csv", out_dir="data/processed_data")
