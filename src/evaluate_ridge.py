import os
import argparse
import json
import pandas as pd
import joblib
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Ridge model and save predictions + metrics")
    parser.add_argument("--X-test", default="data/processed/X_test_scaled.csv")
    parser.add_argument("--y-test", default="data/processed_data/y_test.csv")
    parser.add_argument("--model-path", default="models/model_ridge.pkl")
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--out-predictions", default="data/predictions.csv")
    return parser.parse_args()

def main(args):
    # Chargement
    X_test = pd.read_csv(args.X_test)
    y_test = pd.read_csv(args.y_test)

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()

    model = joblib.load(args.model_path)

    # Prédictions
    y_pred = model.predict(X_test.values)

    # Métriques
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE : {mae:.4f}")
    print(f"Test R2  : {r2:.4f}")

    # Sauvegarde des métriques
    os.makedirs(args.metrics_dir, exist_ok=True)
    scores = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    with open(os.path.join(args.metrics_dir, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2)

    # Sauvegarde des prédictions (y_true, y_pred)
    os.makedirs(os.path.dirname(args.out_predictions), exist_ok=True)
    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })
    pred_df.to_csv(args.out_predictions, index=False)
    print(f"Saved metrics to {os.path.join(args.metrics_dir, 'scores.json')}")
    print(f"Saved predictions to {args.out_predictions}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
