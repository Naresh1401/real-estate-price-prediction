# 🏠 Real Estate Price Prediction & Market Intelligence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ML-powered property valuation platform** with geospatial analytics, market trend analysis, and investment opportunity scoring across 5 major US cities.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Real Estate ML Platform                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐     │
│  │  Raw Data │───▶│   Feature    │───▶│   Model Training   │     │
│  │ 20K Props │    │ Engineering  │    │  8 ML Algorithms   │     │
│  └──────────┘    └──────────────┘    └────────┬───────────┘     │
│                                               │                  │
│                    ┌──────────────────────────┤                  │
│                    ▼                          ▼                  │
│            ┌──────────────┐          ┌───────────────┐          │
│            │  FastAPI      │          │  Streamlit     │          │
│            │  REST API     │          │  Dashboard     │          │
│            │  /predict     │          │  Maps & Charts │          │
│            │  /score       │          │  Valuation     │          │
│            │  /market      │          │  Investment    │          │
│            └──────────────┘          └───────────────┘          │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Investment Scorer: Price Gap · Rental Yield · Growth  │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Cities: NYC · Los Angeles · Chicago · Miami · Austin            │
│  Features: 20+ property & geospatial attributes                  │
│  Models: LinReg · Ridge · Lasso · ElasticNet · RF · XGB · LGBM  │
│          · CatBoost                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Property Valuation**: Predict sale prices using 20+ property and location features
- **Geospatial Analytics**: KMeans clustering on lat/lon, distance-based features, interactive maps
- **Market Intelligence**: City-level market comparisons, neighborhood scoring, trend analysis
- **Investment Scoring**: Identify undervalued properties using predicted vs listed price gap, rental yield estimation, and appreciation potential
- **Model Interpretability**: SHAP values, feature importance, residual analysis
- **REST API**: Production-ready FastAPI endpoints for price prediction and investment scoring
- **Interactive Dashboard**: Streamlit app with maps, charts, and property valuation forms

## Results

| Model | MAE ($) | RMSE ($) | MAPE (%) | R² |
|-------|---------|----------|----------|----|
| Linear Regression | ~45,000 | ~62,000 | ~12.5 | 0.82 |
| Ridge | ~44,800 | ~61,500 | ~12.3 | 0.83 |
| Random Forest | ~28,000 | ~41,000 | ~7.8 | 0.92 |
| XGBoost | ~25,500 | ~37,000 | ~7.1 | 0.94 |
| **LightGBM** | **~24,000** | **~35,500** | **~6.8** | **0.94** |
| CatBoost | ~25,000 | ~36,500 | ~7.0 | 0.94 |

## Quick Start

```bash
# Install dependencies
make install

# Generate synthetic dataset (20K properties)
make data

# Train all models
make train

# Evaluate and generate plots
make evaluate

# Launch API server
make api

# Launch dashboard
make dashboard
```

## Project Structure

```
real-estate-price-prediction/
├── data/
│   ├── raw/                    # Raw generated data
│   └── processed/              # Feature-engineered data
├── models/                     # Saved model artifacts
├── plots/                      # Evaluation plots
├── src/
│   ├── data/
│   │   └── generate_data.py    # Synthetic data generation
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py            # Multi-model training
│   │   ├── evaluate.py         # Model evaluation & plots
│   │   └── investment_scorer.py
│   ├── api/
│   │   └── app.py              # FastAPI endpoints
│   └── dashboard/
│       └── app.py              # Streamlit dashboard
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/price` | Predict property sale price |
| POST | `/score/investment` | Score investment opportunity |
| GET | `/market/{city}` | Get market statistics for a city |
| GET | `/health` | Health check |

## Author

**Naresh Sampangi**

---

*Built with scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, FastAPI, Streamlit, and Plotly.*
