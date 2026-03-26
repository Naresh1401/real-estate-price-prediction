"""
Train multiple ML models for real estate price prediction.
Models: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, XGBoost, LightGBM, CatBoost.
Stratified split by city. Saves the best model.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.features.feature_engineering import load_raw_data, engineer_features

warnings.filterwarnings("ignore")

TARGET = "sale_price"

EXCLUDE_COLS = [
    "sale_price", "log_sale_price", "price_per_sqft", "price_vs_city_median",
    "city", "property_type", "age_bucket",
]


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS and df[c].dtype in [np.float64, np.int64, np.int32, np.float32]]


def compute_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true.clip(lower=1))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2), "R2": round(r2, 4)}


def get_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=100.0, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=100.0, l1_ratio=0.5, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            n_jobs=-1, random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300, max_depth=10, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300, depth=8, learning_rate=0.1,
            random_seed=42, verbose=0,
        ),
    }


def main():
    print("=" * 60)
    print("  Real Estate Price Prediction — Model Training")
    print("=" * 60)

    # Load and engineer features
    print("\n[1/4] Loading and engineering features...")
    raw_df = load_raw_data()
    df = engineer_features(raw_df)
    print(f"  Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Prepare features
    feature_cols = get_feature_columns(df)
    print(f"  Features: {len(feature_cols)}")

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    # Handle missing / infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Stratified split by city
    print("\n[2/4] Splitting data (stratified by city)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["city"],
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index,
    )

    # Train models
    print("\n[3/4] Training models...\n")
    models = get_models()
    results = {}
    best_model_name = None
    best_r2 = -np.inf

    linear_models = {"LinearRegression", "Ridge", "Lasso", "ElasticNet"}

    for name, model in models.items():
        print(f"  Training {name}...", end=" ")
        use_scaled = name in linear_models
        Xtr = X_train_scaled if use_scaled else X_train
        Xte = X_test_scaled if use_scaled else X_test

        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        preds = np.clip(preds, 0, None)  # no negative prices

        metrics = compute_metrics(y_test.values, preds)
        results[name] = metrics

        marker = ""
        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model_name = name
            marker = " ★"

        print(f"MAE=${metrics['MAE']:,.0f}  RMSE=${metrics['RMSE']:,.0f}  "
              f"MAPE={metrics['MAPE']:.1f}%  R²={metrics['R2']:.4f}{marker}")

    # Save best model and artifacts
    print(f"\n[4/4] Saving artifacts...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    best_model = models[best_model_name]
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(feature_cols, "models/feature_columns.joblib")

    with open("models/training_results.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            "results": results,
            "feature_count": len(feature_cols),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }, f, indent=2)

    # Save test predictions for evaluation
    test_df = df.loc[X_test.index].copy()
    test_df["predicted_price"] = best_model.predict(
        X_test_scaled if best_model_name in linear_models else X_test
    )
    test_df["predicted_price"] = test_df["predicted_price"].clip(lower=0).astype(int)
    test_df.to_csv("data/processed/test_predictions.csv", index=False)

    print(f"\n  Best model: {best_model_name} (R²={best_r2:.4f})")
    print(f"  Model saved to models/best_model.joblib")
    print(f"  Results saved to models/training_results.json")
    print(f"  Test predictions saved to data/processed/test_predictions.csv")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
