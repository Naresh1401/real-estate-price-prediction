"""
Feature engineering for real estate price prediction.
Transforms raw property data into ML-ready features with geospatial clustering,
interaction terms, and domain-specific transformations.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures


def load_raw_data(path: str = "data/raw/properties.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def add_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    df["price_per_sqft"] = df["sale_price"] / df["sqft"]
    return df


def add_property_age(df: pd.DataFrame, current_year: int = 2024) -> pd.DataFrame:
    df["property_age"] = current_year - df["year_built"]
    df["is_new_construction"] = (df["property_age"] <= 5).astype(int)
    df["age_bucket"] = pd.cut(
        df["property_age"],
        bins=[0, 5, 15, 30, 50, 125],
        labels=["New", "Recent", "Established", "Mature", "Historic"],
    )
    return df


def add_neighborhood_premium(df: pd.DataFrame) -> pd.DataFrame:
    df["neighborhood_premium"] = (
        df["neighborhood_score"] * 0.4
        + df["school_rating"] * 0.3
        + (10 - df["crime_index"]) * 0.3
    )
    df["walkability_index"] = (df["walk_score"] + df["transit_score"]) / 2
    df["location_quality"] = (
        df["neighborhood_premium"] * 0.5
        + df["walkability_index"] / 10 * 0.3
        + (10 - df["distance_to_downtown"].clip(upper=30) / 3) * 0.2
    )
    return df


def add_geospatial_clusters(df: pd.DataFrame, n_clusters: int = 25) -> pd.DataFrame:
    coords = df[["latitude", "longitude"]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["geo_cluster"] = kmeans.fit_predict(coords)

    cluster_stats = df.groupby("geo_cluster")["sale_price"].agg(["mean", "median", "std"])
    cluster_stats.columns = ["cluster_avg_price", "cluster_median_price", "cluster_price_std"]
    df = df.merge(cluster_stats, left_on="geo_cluster", right_index=True, how="left")

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["sqft_x_bedrooms"] = df["sqft"] * df["bedrooms"]
    df["sqft_x_bathrooms"] = df["sqft"] * df["bathrooms"]
    df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"].clip(lower=1)
    df["sqft_per_bedroom"] = df["sqft"] / df["bedrooms"].clip(lower=1)
    df["lot_to_sqft_ratio"] = df["lot_size"] / df["sqft"].clip(lower=1)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["amenity_score"] = df["garage"] * 0.4 + df["pool"] * 0.6
    return df


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["sqft", "lot_size", "distance_to_downtown", "median_income_area"]:
        df[f"log_{col}"] = np.log1p(df[col])
    df["log_sale_price"] = np.log1p(df["sale_price"])
    return df


def add_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    poly_cols = ["sqft", "property_age", "neighborhood_premium"]
    available = [c for c in poly_cols if c in df.columns]
    if not available:
        return df

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_data = poly.fit_transform(df[available])
    poly_names = poly.get_feature_names_out(available)

    new_cols = [n for n in poly_names if n not in available]
    new_data = poly_data[:, len(available):]
    for i, name in enumerate(new_cols):
        clean_name = name.replace(" ", "_")
        df[f"poly_{clean_name}"] = new_data[:, i]

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    le_city = LabelEncoder()
    df["city_encoded"] = le_city.fit_transform(df["city"])

    le_type = LabelEncoder()
    df["property_type_encoded"] = le_type.fit_transform(df["property_type"])

    city_dummies = pd.get_dummies(df["city"], prefix="city", dtype=int)
    type_dummies = pd.get_dummies(df["property_type"], prefix="type", dtype=int)
    df = pd.concat([df, city_dummies, type_dummies], axis=1)

    return df


def add_city_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    city_medians = df.groupby("city")["sale_price"].transform("median")
    df["price_vs_city_median"] = df["sale_price"] / city_medians

    city_sqft_medians = df.groupby("city")["sqft"].transform("median")
    df["sqft_vs_city_median"] = df["sqft"] / city_sqft_medians

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_price_per_sqft(df)
    df = add_property_age(df)
    df = add_neighborhood_premium(df)
    df = add_geospatial_clusters(df)
    df = add_interaction_features(df)
    df = add_log_transforms(df)
    df = add_polynomial_features(df)
    df = encode_categoricals(df)
    df = add_city_relative_features(df)
    return df


def main():
    print("Loading raw data...")
    df = load_raw_data()
    print(f"  Raw shape: {df.shape}")

    print("Engineering features...")
    df = engineer_features(df)
    print(f"  Engineered shape: {df.shape}")

    os.makedirs("data/processed", exist_ok=True)
    save_path = "data/processed/properties_featured.csv"
    df.to_csv(save_path, index=False)
    print(f"  Saved to {save_path}")

    print(f"\nFeature columns ({df.shape[1]} total):")
    for col in sorted(df.columns):
        print(f"    {col}")


if __name__ == "__main__":
    main()
