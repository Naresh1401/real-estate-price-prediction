"""
FastAPI application for real estate price prediction and investment scoring.
Endpoints: POST /predict/price, POST /score/investment, GET /market/{city}, GET /health
"""

import os
import json
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.feature_engineering import engineer_features
from src.models.investment_scorer import score_investment

app = FastAPI(
    title="Real Estate Price Prediction API",
    description="ML-powered property valuation, market intelligence, and investment scoring.",
    version="1.0.0",
)

# ---------- Lazy-loaded globals ----------
_model = None
_scaler = None
_feature_cols = None
_training_results = None
_raw_data = None


def _load_model():
    global _model, _scaler, _feature_cols, _training_results
    if _model is None:
        _model = joblib.load("models/best_model.joblib")
        _scaler = joblib.load("models/scaler.joblib")
        _feature_cols = joblib.load("models/feature_columns.joblib")
        with open("models/training_results.json") as f:
            _training_results = json.load(f)


def _load_data():
    global _raw_data
    if _raw_data is None:
        _raw_data = pd.read_csv("data/raw/properties.csv")


# ---------- Request / Response schemas ----------

class PropertyInput(BaseModel):
    city: str = Field(..., example="Austin")
    property_type: str = Field(..., example="Single Family")
    sqft: int = Field(..., ge=200, le=15000, example=2000)
    bedrooms: int = Field(..., ge=1, le=10, example=3)
    bathrooms: float = Field(..., ge=1, le=8, example=2.5)
    year_built: int = Field(..., ge=1800, le=2025, example=2005)
    lot_size: int = Field(..., ge=0, le=100000, example=5000)
    neighborhood_score: float = Field(..., ge=1, le=10, example=7.5)
    school_rating: float = Field(..., ge=1, le=10, example=7.0)
    crime_index: float = Field(..., ge=1, le=10, example=3.0)
    walk_score: int = Field(..., ge=0, le=100, example=72)
    transit_score: int = Field(..., ge=0, le=100, example=55)
    distance_to_downtown: float = Field(..., ge=0, le=100, example=5.0)
    distance_to_parks: float = Field(..., ge=0, le=30, example=1.2)
    latitude: float = Field(..., example=30.2672)
    longitude: float = Field(..., example=-97.7431)
    HOA_fees: int = Field(0, ge=0, le=5000, example=150)
    garage: int = Field(0, ge=0, le=5, example=2)
    pool: int = Field(0, ge=0, le=1, example=0)
    property_tax_rate: float = Field(1.2, ge=0.1, le=5.0, example=1.2)
    median_income_area: int = Field(60000, ge=15000, le=500000, example=72000)
    population_density: int = Field(5000, ge=100, le=100000, example=3300)


class InvestmentInput(PropertyInput):
    listed_price: float = Field(..., ge=10000, example=450000)


class PricePrediction(BaseModel):
    predicted_price: float
    confidence_band_low: float
    confidence_band_high: float
    price_per_sqft: float
    model_used: str


class InvestmentResult(BaseModel):
    predicted_price: float
    overall_score: float
    overall_rating: str
    price_analysis: dict
    rental_analysis: dict
    appreciation_analysis: dict


# ---------- Helpers ----------

def _property_to_df(prop: PropertyInput) -> pd.DataFrame:
    row = prop.model_dump()
    row["sale_price"] = 0  # placeholder — excluded from model features via EXCLUDE_COLS
    return pd.DataFrame([row])


def _predict_price(prop: PropertyInput) -> float:
    _load_model()
    _load_data()
    df = _property_to_df(prop)

    # Append to training data for accurate geospatial clustering & city-relative features
    full_df = pd.concat([_raw_data, df], ignore_index=True)
    full_df = engineer_features(full_df)

    # Extract only the new property (last row) with engineered features
    df_engineered = full_df.iloc[[-1]]

    X = df_engineered[_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Use scaler only for linear models
    linear_models = {"LinearRegression", "Ridge", "Lasso", "ElasticNet"}
    if _training_results["best_model"] in linear_models:
        X = pd.DataFrame(_scaler.transform(X), columns=_feature_cols)

    pred = float(_model.predict(X)[0])
    return max(pred, 10000)


# ---------- Endpoints ----------

@app.get("/health")
def health_check():
    model_loaded = _model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "version": "1.0.0",
    }


@app.post("/predict/price", response_model=PricePrediction)
def predict_price(prop: PropertyInput):
    try:
        _load_model()
        predicted = _predict_price(prop)
        band = predicted * 0.08  # ±8 % confidence band
        return PricePrediction(
            predicted_price=round(predicted, 2),
            confidence_band_low=round(predicted - band, 2),
            confidence_band_high=round(predicted + band, 2),
            price_per_sqft=round(predicted / max(prop.sqft, 1), 2),
            model_used=_training_results["best_model"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/investment", response_model=InvestmentResult)
def score_investment_endpoint(inp: InvestmentInput):
    try:
        predicted = _predict_price(inp)
        result = score_investment(
            predicted_price=predicted,
            listed_price=inp.listed_price,
            city=inp.city,
            sqft=inp.sqft,
            bedrooms=inp.bedrooms,
            neighborhood_score=inp.neighborhood_score,
            property_age=2024 - inp.year_built,
            population_density=inp.population_density,
            median_income_area=inp.median_income_area,
        )
        return InvestmentResult(
            predicted_price=round(predicted, 2),
            overall_score=result["overall_score"],
            overall_rating=result["overall_rating"],
            price_analysis=result["price_analysis"],
            rental_analysis=result["rental_analysis"],
            appreciation_analysis=result["appreciation_analysis"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market/{city}")
def get_market_stats(city: str):
    _load_data()
    valid_cities = _raw_data["city"].unique().tolist()

    if city not in valid_cities:
        raise HTTPException(
            status_code=404,
            detail=f"City '{city}' not found. Available: {valid_cities}",
        )

    subset = _raw_data[_raw_data["city"] == city]

    stats = {
        "city": city,
        "total_properties": int(len(subset)),
        "price_stats": {
            "mean": round(float(subset["sale_price"].mean()), 2),
            "median": round(float(subset["sale_price"].median()), 2),
            "min": int(subset["sale_price"].min()),
            "max": int(subset["sale_price"].max()),
            "std": round(float(subset["sale_price"].std()), 2),
        },
        "sqft_stats": {
            "mean": round(float(subset["sqft"].mean()), 2),
            "median": round(float(subset["sqft"].median()), 2),
        },
        "property_type_breakdown": subset["property_type"].value_counts().to_dict(),
        "avg_neighborhood_score": round(float(subset["neighborhood_score"].mean()), 2),
        "avg_school_rating": round(float(subset["school_rating"].mean()), 2),
        "median_price_per_sqft": round(
            float((subset["sale_price"] / subset["sqft"]).median()), 2,
        ),
    }
    return stats
