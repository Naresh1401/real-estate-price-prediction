"""
Investment opportunity scoring based on predicted vs listed price gap,
rental yield estimation, and appreciation potential.
"""

import numpy as np
import pandas as pd


# City-level rental yield benchmarks (annual gross yield %)
RENTAL_YIELD_BENCHMARKS = {
    "New York": 4.2,
    "Los Angeles": 4.0,
    "Chicago": 6.5,
    "Miami": 5.5,
    "Austin": 5.0,
}

# City-level annual appreciation rates (%)
APPRECIATION_RATES = {
    "New York": 3.5,
    "Los Angeles": 4.0,
    "Chicago": 2.5,
    "Miami": 5.0,
    "Austin": 6.0,
}


def compute_price_gap_score(predicted_price: float, listed_price: float) -> dict:
    """Score based on the gap between predicted (fair) value and listed price."""
    gap = predicted_price - listed_price
    gap_pct = (gap / listed_price) * 100

    if gap_pct > 15:
        rating = "Strong Buy"
        score = min(10, 7 + gap_pct * 0.1)
    elif gap_pct > 5:
        rating = "Buy"
        score = 5 + gap_pct * 0.15
    elif gap_pct > -5:
        rating = "Fair Value"
        score = 5
    elif gap_pct > -15:
        rating = "Overpriced"
        score = max(2, 5 + gap_pct * 0.15)
    else:
        rating = "Avoid"
        score = max(0, 2 + gap_pct * 0.05)

    return {
        "price_gap": round(gap, 2),
        "price_gap_pct": round(gap_pct, 2),
        "price_gap_rating": rating,
        "price_gap_score": round(np.clip(score, 0, 10), 2),
    }


def estimate_rental_yield(
    predicted_price: float,
    city: str,
    sqft: float,
    bedrooms: int,
    neighborhood_score: float,
) -> dict:
    """Estimate rental yield based on property characteristics and city benchmarks."""
    base_yield = RENTAL_YIELD_BENCHMARKS.get(city, 5.0)

    # Adjustments
    size_adj = -0.3 if sqft > 2500 else (0.3 if sqft < 1000 else 0)
    bed_adj = 0.2 * max(0, bedrooms - 2)
    hood_adj = (neighborhood_score - 5) * 0.15

    adjusted_yield = base_yield + size_adj + bed_adj + hood_adj
    adjusted_yield = np.clip(adjusted_yield, 2.0, 12.0)

    monthly_rent = (predicted_price * adjusted_yield / 100) / 12
    annual_rent = monthly_rent * 12

    return {
        "estimated_monthly_rent": round(monthly_rent, 2),
        "estimated_annual_rent": round(annual_rent, 2),
        "gross_rental_yield_pct": round(adjusted_yield, 2),
        "rental_score": round(np.clip(adjusted_yield / 1.2, 0, 10), 2),
    }


def estimate_appreciation(
    city: str,
    neighborhood_score: float,
    property_age: int,
    population_density: float,
    median_income_area: float,
) -> dict:
    """Estimate appreciation potential based on neighborhood trends."""
    base_rate = APPRECIATION_RATES.get(city, 3.5)

    hood_adj = (neighborhood_score - 5) * 0.3
    age_adj = 0.5 if property_age < 10 else (-0.3 if property_age > 50 else 0)
    density_adj = 0.2 if population_density > 10000 else (-0.1 if population_density < 2000 else 0)
    income_adj = 0.3 if median_income_area > 80000 else (-0.2 if median_income_area < 40000 else 0)

    appreciation_rate = base_rate + hood_adj + age_adj + density_adj + income_adj
    appreciation_rate = np.clip(appreciation_rate, 0.5, 12.0)

    return {
        "estimated_annual_appreciation_pct": round(appreciation_rate, 2),
        "5yr_projected_gain_pct": round((1 + appreciation_rate / 100) ** 5 * 100 - 100, 2),
        "appreciation_score": round(np.clip(appreciation_rate / 1.0, 0, 10), 2),
    }


def score_investment(
    predicted_price: float,
    listed_price: float,
    city: str,
    sqft: float,
    bedrooms: int,
    neighborhood_score: float,
    property_age: int,
    population_density: float,
    median_income_area: float,
) -> dict:
    """Compute overall investment score combining all factors."""

    gap = compute_price_gap_score(predicted_price, listed_price)
    rental = estimate_rental_yield(predicted_price, city, sqft, bedrooms, neighborhood_score)
    appreciation = estimate_appreciation(
        city, neighborhood_score, property_age, population_density, median_income_area,
    )

    overall_score = (
        gap["price_gap_score"] * 0.40
        + rental["rental_score"] * 0.30
        + appreciation["appreciation_score"] * 0.30
    )
    overall_score = round(np.clip(overall_score, 0, 10), 2)

    if overall_score >= 8:
        overall_rating = "Excellent Investment"
    elif overall_score >= 6.5:
        overall_rating = "Good Investment"
    elif overall_score >= 5:
        overall_rating = "Average"
    elif overall_score >= 3:
        overall_rating = "Below Average"
    else:
        overall_rating = "Poor Investment"

    return {
        "overall_score": overall_score,
        "overall_rating": overall_rating,
        "price_analysis": gap,
        "rental_analysis": rental,
        "appreciation_analysis": appreciation,
    }


def score_dataframe(df: pd.DataFrame, predicted_col: str = "predicted_price") -> pd.DataFrame:
    """Score all properties in a DataFrame."""
    scores = []
    for _, row in df.iterrows():
        result = score_investment(
            predicted_price=row[predicted_col],
            listed_price=row.get("sale_price", row[predicted_col]),
            city=row["city"],
            sqft=row["sqft"],
            bedrooms=row["bedrooms"],
            neighborhood_score=row["neighborhood_score"],
            property_age=2024 - row["year_built"],
            population_density=row["population_density"],
            median_income_area=row["median_income_area"],
        )
        scores.append({
            "overall_score": result["overall_score"],
            "overall_rating": result["overall_rating"],
            "price_gap_pct": result["price_analysis"]["price_gap_pct"],
            "rental_yield_pct": result["rental_analysis"]["gross_rental_yield_pct"],
            "appreciation_pct": result["appreciation_analysis"]["estimated_annual_appreciation_pct"],
            "monthly_rent_est": result["rental_analysis"]["estimated_monthly_rent"],
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(scores)], axis=1)
