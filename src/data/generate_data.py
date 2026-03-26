"""
Generate synthetic real estate dataset with 20,000 properties across 5 US cities.
Realistic price distributions based on city, property features, and location attributes.
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N_PROPERTIES = 20000

CITY_CONFIG = {
    "New York": {
        "count": 5000,
        "lat_center": 40.7128, "lon_center": -74.0060,
        "lat_std": 0.05, "lon_std": 0.04,
        "base_price": 750_000, "price_std": 300_000,
        "sqft_mean": 1200, "sqft_std": 600,
        "property_types": ["Condo", "Co-op", "Townhouse", "Single Family"],
        "type_weights": [0.40, 0.25, 0.20, 0.15],
        "median_income": 85_000, "pop_density": 27_000,
    },
    "Los Angeles": {
        "count": 4500,
        "lat_center": 34.0522, "lon_center": -118.2437,
        "lat_std": 0.08, "lon_std": 0.10,
        "base_price": 850_000, "price_std": 400_000,
        "sqft_mean": 1800, "sqft_std": 800,
        "property_types": ["Single Family", "Condo", "Townhouse", "Multi-Family"],
        "type_weights": [0.45, 0.30, 0.15, 0.10],
        "median_income": 75_000, "pop_density": 8_000,
    },
    "Chicago": {
        "count": 4000,
        "lat_center": 41.8781, "lon_center": -87.6298,
        "lat_std": 0.06, "lon_std": 0.05,
        "base_price": 350_000, "price_std": 180_000,
        "sqft_mean": 1500, "sqft_std": 700,
        "property_types": ["Single Family", "Condo", "Townhouse", "Multi-Family"],
        "type_weights": [0.35, 0.30, 0.20, 0.15],
        "median_income": 62_000, "pop_density": 12_000,
    },
    "Miami": {
        "count": 3500,
        "lat_center": 25.7617, "lon_center": -80.1918,
        "lat_std": 0.06, "lon_std": 0.06,
        "base_price": 500_000, "price_std": 250_000,
        "sqft_mean": 1600, "sqft_std": 700,
        "property_types": ["Condo", "Single Family", "Townhouse", "Multi-Family"],
        "type_weights": [0.40, 0.30, 0.20, 0.10],
        "median_income": 55_000, "pop_density": 12_500,
    },
    "Austin": {
        "count": 3000,
        "lat_center": 30.2672, "lon_center": -97.7431,
        "lat_std": 0.07, "lon_std": 0.07,
        "base_price": 450_000, "price_std": 200_000,
        "sqft_mean": 2000, "sqft_std": 800,
        "property_types": ["Single Family", "Condo", "Townhouse", "Multi-Family"],
        "type_weights": [0.50, 0.25, 0.15, 0.10],
        "median_income": 72_000, "pop_density": 3_300,
    },
}

PROPERTY_TYPE_MULTIPLIER = {
    "Single Family": 1.15,
    "Condo": 0.90,
    "Co-op": 0.85,
    "Townhouse": 1.05,
    "Multi-Family": 1.25,
}


def generate_city_data(city: str, config: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = config["count"]

    property_type = rng.choice(
        config["property_types"], size=n, p=config["type_weights"]
    )

    sqft = np.clip(
        rng.normal(config["sqft_mean"], config["sqft_std"], n), 400, 8000
    ).astype(int)

    bedrooms = np.clip(np.round(sqft / 500 + rng.normal(0, 0.5, n)), 1, 8).astype(int)
    bathrooms = np.clip(
        np.round(bedrooms * 0.75 + rng.normal(0, 0.3, n) + 0.5) * 0.5, 1, 6
    )
    bathrooms = (bathrooms * 2).astype(int) / 2  # round to 0.5

    year_built = np.clip(
        rng.normal(1985, 20, n), 1900, 2024
    ).astype(int)

    lot_size = np.where(
        np.isin(property_type, ["Single Family", "Multi-Family"]),
        np.clip(rng.normal(sqft * 2.5, sqft * 0.8, n), 1000, 40000),
        np.clip(rng.normal(sqft * 0.3, sqft * 0.1, n), 0, 5000),
    ).astype(int)

    latitude = rng.normal(config["lat_center"], config["lat_std"], n)
    longitude = rng.normal(config["lon_center"], config["lon_std"], n)

    neighborhood_score = np.clip(rng.normal(6.5, 1.8, n), 1, 10).round(1)
    school_rating = np.clip(rng.normal(6.0, 2.0, n), 1, 10).round(1)
    crime_index = np.clip(10 - neighborhood_score + rng.normal(0, 1.2, n), 1, 10).round(1)
    walk_score = np.clip(rng.normal(65, 20, n), 0, 100).astype(int)
    transit_score = np.clip(rng.normal(55, 25, n), 0, 100).astype(int)

    dist_downtown = np.sqrt(
        (latitude - config["lat_center"]) ** 2
        + (longitude - config["lon_center"]) ** 2
    ) * 111  # rough km conversion
    distance_to_downtown = np.clip(dist_downtown + rng.normal(0, 1, n), 0.5, 50).round(2)

    distance_to_parks = np.clip(rng.exponential(1.5, n), 0.1, 15).round(2)

    hoa_fees = np.where(
        np.isin(property_type, ["Condo", "Co-op"]),
        np.clip(rng.normal(450, 200, n), 50, 2000),
        np.where(
            property_type == "Townhouse",
            np.clip(rng.normal(200, 100, n), 0, 800),
            rng.choice([0, 0, 0, 50, 100], size=n),
        ),
    ).astype(int)

    garage = rng.choice([0, 1, 2, 3], size=n, p=[0.25, 0.35, 0.30, 0.10])
    pool = rng.choice(
        [0, 1], size=n,
        p=[0.7, 0.3] if city in ["Miami", "Los Angeles", "Austin"] else [0.85, 0.15],
    )

    property_tax_rate = np.clip(rng.normal(1.2, 0.3, n), 0.5, 3.0).round(3)

    median_income_area = np.clip(
        rng.normal(config["median_income"], config["median_income"] * 0.3, n),
        25_000, 250_000,
    ).astype(int)

    population_density = np.clip(
        rng.normal(config["pop_density"], config["pop_density"] * 0.4, n),
        500, 70_000,
    ).astype(int)

    # --- Compute realistic sale_price ---
    type_mult = np.array([PROPERTY_TYPE_MULTIPLIER[t] for t in property_type])

    sqft_factor = (sqft / config["sqft_mean"]) ** 0.85
    age_factor = 1 + (year_built - 1960) * 0.002
    neighborhood_factor = 1 + (neighborhood_score - 5) * 0.06
    school_factor = 1 + (school_rating - 5) * 0.03
    crime_penalty = 1 - (crime_index - 5) * 0.02
    downtown_factor = 1 - np.log1p(distance_to_downtown) * 0.04
    income_factor = (median_income_area / config["median_income"]) ** 0.3
    garage_bonus = 1 + garage * 0.03
    pool_bonus = 1 + pool * 0.04

    price = (
        config["base_price"]
        * type_mult
        * sqft_factor
        * age_factor
        * neighborhood_factor
        * school_factor
        * crime_penalty
        * downtown_factor
        * income_factor
        * garage_bonus
        * pool_bonus
    )

    noise = rng.normal(1.0, 0.08, n)
    price = (price * noise).astype(int)
    price = np.clip(price, 50_000, 15_000_000)

    return pd.DataFrame({
        "city": city,
        "property_type": property_type,
        "sqft": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "lot_size": lot_size,
        "neighborhood_score": neighborhood_score,
        "school_rating": school_rating,
        "crime_index": crime_index,
        "walk_score": walk_score,
        "transit_score": transit_score,
        "distance_to_downtown": distance_to_downtown,
        "distance_to_parks": distance_to_parks,
        "latitude": latitude.round(6),
        "longitude": longitude.round(6),
        "HOA_fees": hoa_fees,
        "garage": garage,
        "pool": pool,
        "property_tax_rate": property_tax_rate,
        "median_income_area": median_income_area,
        "population_density": population_density,
        "sale_price": price,
    })


def main():
    rng = np.random.default_rng(SEED)

    frames = []
    for city, config in CITY_CONFIG.items():
        print(f"Generating {config['count']} properties for {city}...")
        df = generate_city_data(city, config, rng)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    os.makedirs("data/raw", exist_ok=True)
    save_path = "data/raw/properties.csv"
    data.to_csv(save_path, index=False)

    print(f"\nDataset saved to {save_path}")
    print(f"  Total properties: {len(data):,}")
    print(f"  Cities: {data['city'].nunique()}")
    print(f"  Columns: {data.shape[1]}")
    print(f"\nPrice summary by city:")
    print(data.groupby("city")["sale_price"].describe()[["mean", "50%", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
