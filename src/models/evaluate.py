"""
Model evaluation: actual vs predicted scatter, residual analysis,
feature importance, SHAP values, error analysis by city/property type.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

PLOTS_DIR = "plots"


def load_artifacts():
    model = joblib.load("models/best_model.joblib")
    feature_cols = joblib.load("models/feature_columns.joblib")
    test_df = pd.read_csv("data/processed/test_predictions.csv")

    with open("models/training_results.json") as f:
        results = json.load(f)

    return model, feature_cols, test_df, results


def plot_actual_vs_predicted(test_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        test_df["sale_price"], test_df["predicted_price"],
        c=test_df["city"].astype("category").cat.codes,
        alpha=0.3, s=10, cmap="Set1",
    )
    lims = [
        min(test_df["sale_price"].min(), test_df["predicted_price"].min()),
        max(test_df["sale_price"].max(), test_df["predicted_price"].max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Price ($)", fontsize=12)
    ax.set_ylabel("Predicted Price ($)", fontsize=12)
    ax.set_title("Actual vs Predicted Sale Price", fontsize=14)
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/actual_vs_predicted.png", dpi=150)
    plt.close()


def plot_residuals(test_df: pd.DataFrame):
    residuals = test_df["sale_price"] - test_df["predicted_price"]
    pct_error = residuals / test_df["sale_price"] * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Residual distribution
    axes[0].hist(residuals, bins=80, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Residual ($)")
    axes[0].set_title("Residual Distribution")

    # Residual vs predicted
    axes[1].scatter(test_df["predicted_price"], residuals, alpha=0.2, s=8)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted Price ($)")
    axes[1].set_ylabel("Residual ($)")
    axes[1].set_title("Residuals vs Predicted")

    # Percentage error distribution
    axes[2].hist(pct_error.clip(-50, 50), bins=80, edgecolor="black", alpha=0.7, color="coral")
    axes[2].axvline(0, color="red", linestyle="--")
    axes[2].set_xlabel("Percentage Error (%)")
    axes[2].set_title("Percentage Error Distribution")

    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/residual_analysis.png", dpi=150)
    plt.close()


def plot_feature_importance(model, feature_cols: list):
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_)

    if importance is None:
        return

    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=True)
    top_n = feat_imp.tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n.plot(kind="barh", ax=ax, color="teal")
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances")
    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=150)
    plt.close()


def plot_shap_summary(model, X_test: pd.DataFrame, feature_cols: list):
    try:
        import shap
    except ImportError:
        print("  SHAP not installed, skipping SHAP plots.")
        return

    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_test)

        sample = X_test.sample(min(500, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(sample)

        fig, ax = plt.subplots(figsize=(10, 10))
        shap.summary_plot(shap_values, sample, feature_names=feature_cols, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        print(f"  SHAP analysis skipped: {e}")


def plot_error_by_city(test_df: pd.DataFrame):
    test_df["abs_error"] = np.abs(test_df["sale_price"] - test_df["predicted_price"])
    test_df["pct_error"] = test_df["abs_error"] / test_df["sale_price"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    city_mae = test_df.groupby("city")["abs_error"].mean().sort_values()
    city_mae.plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_xlabel("Mean Absolute Error ($)")
    axes[0].set_title("MAE by City")

    city_mape = test_df.groupby("city")["pct_error"].mean().sort_values()
    city_mape.plot(kind="barh", ax=axes[1], color="coral")
    axes[1].set_xlabel("Mean Absolute Percentage Error (%)")
    axes[1].set_title("MAPE by City")

    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/error_by_city.png", dpi=150)
    plt.close()


def plot_error_by_property_type(test_df: pd.DataFrame):
    if "abs_error" not in test_df.columns:
        test_df["abs_error"] = np.abs(test_df["sale_price"] - test_df["predicted_price"])
        test_df["pct_error"] = test_df["abs_error"] / test_df["sale_price"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    type_mae = test_df.groupby("property_type")["abs_error"].mean().sort_values()
    type_mae.plot(kind="barh", ax=axes[0], color="teal")
    axes[0].set_xlabel("Mean Absolute Error ($)")
    axes[0].set_title("MAE by Property Type")

    type_mape = test_df.groupby("property_type")["pct_error"].mean().sort_values()
    type_mape.plot(kind="barh", ax=axes[1], color="salmon")
    axes[1].set_xlabel("MAPE (%)")
    axes[1].set_title("MAPE by Property Type")

    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/error_by_property_type.png", dpi=150)
    plt.close()


def plot_market_segments(test_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Price distribution by city
    for city in test_df["city"].unique():
        subset = test_df[test_df["city"] == city]
        axes[0, 0].hist(subset["sale_price"], bins=40, alpha=0.5, label=city)
    axes[0, 0].set_xlabel("Sale Price ($)")
    axes[0, 0].set_title("Price Distribution by City")
    axes[0, 0].legend()

    # Median price by city + property type
    pivot = test_df.pivot_table(values="sale_price", index="city", columns="property_type", aggfunc="median")
    pivot.plot(kind="bar", ax=axes[0, 1])
    axes[0, 1].set_title("Median Price by City & Type")
    axes[0, 1].set_ylabel("Median Price ($)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Sqft vs price
    axes[1, 0].scatter(test_df["sqft"], test_df["sale_price"], alpha=0.15, s=5)
    axes[1, 0].set_xlabel("Square Footage")
    axes[1, 0].set_ylabel("Sale Price ($)")
    axes[1, 0].set_title("Sqft vs Price")

    # Model comparison
    with open("models/training_results.json") as f:
        results = json.load(f)["results"]
    model_names = list(results.keys())
    r2_scores = [results[m]["R2"] for m in model_names]
    colors = ["gold" if s == max(r2_scores) else "steelblue" for s in r2_scores]
    axes[1, 1].barh(model_names, r2_scores, color=colors)
    axes[1, 1].set_xlabel("R² Score")
    axes[1, 1].set_title("Model Comparison")
    axes[1, 1].set_xlim(min(r2_scores) - 0.05, 1.0)

    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/market_segments.png", dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("  Real Estate Price Prediction — Model Evaluation")
    print("=" * 60)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\nLoading artifacts...")
    model, feature_cols, test_df, results = load_artifacts()
    print(f"  Best model: {results['best_model']}")
    print(f"  Test samples: {len(test_df):,}")

    print("\nGenerating plots...")

    print("  [1/7] Actual vs Predicted...")
    plot_actual_vs_predicted(test_df)

    print("  [2/7] Residual Analysis...")
    plot_residuals(test_df)

    print("  [3/7] Feature Importance...")
    plot_feature_importance(model, feature_cols)

    print("  [4/7] SHAP Summary...")
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    plot_shap_summary(model, X_test, feature_cols)

    print("  [5/7] Error by City...")
    plot_error_by_city(test_df)

    print("  [6/7] Error by Property Type...")
    plot_error_by_property_type(test_df)

    print("  [7/7] Market Segments...")
    plot_market_segments(test_df)

    print(f"\nAll plots saved to {PLOTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
