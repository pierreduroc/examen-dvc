import os
import argparse
import pickle
import pandas as pd
from sklearn.linear_model import Ridge
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description="Train Ridge model using best params from param.pkl")
    parser.add_argument("--X-train", default="data/processed/X_train_scaled.csv")
    parser.add_argument("--y-train", default="data/processed_data/y_train.csv")
    parser.add_argument("--params-path", default="models/param.pkl")
    parser.add_argument("--out-model", default="models/model_ridge.pkl")
    return parser.parse_args()

def main(args):
    # Chargement des données
    X_train = pd.read_csv(args.X_train)
    y_train = pd.read_csv(args.y_train)

    # Chargement des meilleurs hyperparamètres
    with open(args.params_path, "rb") as f:
        best_params = pickle.load(f)
    print(f"Best params loaded: {best_params}")

    # Entraînement du modèle Ridge
    model = Ridge(**best_params)
    model.fit(X_train.values, y_train)

    # Sauvegarde du modèle
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    joblib.dump(model, args.out_model)
    print(f"Model saved to {args.out_model}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
