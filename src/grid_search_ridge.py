import argparse
import os
import pickle
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import root_mean_squared_error, r2_score

def parse_args():
    parser = argparse.ArgumentParser(description="GridSearch Ridge on scaled datasets")
    parser.add_argument("--X-train", dest="X_train", default="data/processed/X_train_scaled.csv")
    parser.add_argument("--X-test", dest="X_test", default="data/processed/X_test_scaled.csv")
    parser.add_argument("--y-train", dest="y_train", default="data/processed_data/y_train.csv")
    parser.add_argument("--y-test", dest="y_test", default="data/processed_data/y_test.csv")
    parser.add_argument("--out-params", dest="out_params", default="models/param.pkl")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main(args):
    # Chargement des données
    X_train = pd.read_csv(args.X_train).values
    X_test  = pd.read_csv(args.X_test).values
    y_train = pd.read_csv(args.y_train).values
    y_test  = pd.read_csv(args.y_test).values

    # Modèle + grille simple adaptée à des features déjà normalisées
    model = Ridge(random_state=args.seed)
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
        'fit_intercept': [True, False]
    }

    # cv & gs
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',  # minimise RMSE
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )
    gs.fit(X_train, y_train)

    # Évaluation sur test
    y_pred = gs.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Logs console
    print("Best params:", gs.best_params_)
    print("Best CV RMSE:", -gs.best_score_)
    print("Test RMSE:", rmse)
    print("Test R2:", r2)

    # Sauvegarde des meilleurs hyperparamètres
    os.makedirs(os.path.dirname(args.out_params), exist_ok=True)
    with open(args.out_params, 'wb') as f:
        pickle.dump(gs.best_params_, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)

