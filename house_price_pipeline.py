
"""
Main pipeline for feature engineering and baseline model training.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from feature_engineering import basic_preprocess, encode_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT  # expects train.csv & test.csv in repo root
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def rmse_cv(model, X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf, n_jobs=-1))
    return rmse

def main():
    print("Loading data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    # Preprocess
    train_p = basic_preprocess(train, is_train=True)
    test_p = basic_preprocess(test, is_train=False)
    train_ids = train_p["Id"].values
    test_ids = test_p["Id"].values
    train_p_nod = train_p.drop(columns=["Id"], errors="ignore")
    test_p_nod = test_p.drop(columns=["Id"], errors="ignore")
    X_train, X_test, y = encode_features(train_p_nod, test_p_nod)
    y = np.log1p(y)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    # Impute
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    # Model
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    scores = rmse_cv(model, X_train_imputed, y, n_folds=5)
    print("CV RMSE (log-target):", scores.mean(), "std:", scores.std())
    model.fit(X_train_imputed, y)
    # Feature importances
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X_train_imputed.columns).sort_values(ascending=False).head(50)
    plt.figure(figsize=(8,10))
    feat_imp.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importances.png", dpi=200)
    joblib.dump(model, OUT_DIR / "rf_model.joblib")
    joblib.dump(imputer, OUT_DIR / "imputer.joblib")
    preds_log = model.predict(X_test_imputed)
    preds = np.expm1(preds_log)
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": preds})
    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    print("Saved submission to", OUT_DIR / "submission.csv")

if __name__ == "__main__":
    main()
