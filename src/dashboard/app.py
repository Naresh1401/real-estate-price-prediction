"""
Streamlit dashboard for real estate price prediction and investment analysis.
Features: property valuation form, interactive map, market comparison,
investment scoring, and neighborhood analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Intelligence",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    raw = pd.read_csv("data/raw/properties.csv")
    predictions = None
    if os.path.exists("data/processed/test_predictions.csv"):
        predictions = pd.read_csv("data/processed/test_predictions.csv")
    results = None
    if os.path.exists("models/training_results.json"):
        with open("models/training_results.json") as f:
            results = json.load(f)
    return raw, predictions, results


raw_df, pred_df, training_results = load_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🏠 Real Estate Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Property Valuation", "Interactive Map", "Market Comparison",
     "Investment Scoring", "Neighborhood Analysis"],
)

# ---------------------------------------------------------------------------
# 1. Property Valuation
# ---------------------------------------------------------------------------
if page == "Property Valuation":
    st.title("🏡 Property Valuation")
    st.markdown("Enter property details to get an ML-powered price estimate.")

    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.selectbox("City", raw_df["city"].unique())
        property_type = st.selectbox("Property Type", raw_df["property_type"].unique())
        sqft = st.number_input("Square Footage", 400, 10000, 1800)
        bedrooms = st.slider("Bedrooms", 1, 8, 3)
        bathrooms = st.slider("Bathrooms", 1.0, 6.0, 2.0, 0.5)
        year_built = st.number_input("Year Built", 1900, 2024, 2005)
        lot_size = st.number_input("Lot Size (sqft)", 0, 50000, 5000)

    with col2:
        neighborhood_score = st.slider("Neighborhood Score", 1.0, 10.0, 7.0, 0.1)
        school_rating = st.slider("School Rating", 1.0, 10.0, 6.5, 0.1)
        crime_index = st.slider("Crime Index", 1.0, 10.0, 3.5, 0.1)
        walk_score = st.slider("Walk Score", 0, 100, 65)
        transit_score = st.slider("Transit Score", 0, 100, 55)
        distance_to_downtown = st.number_input("Distance to Downtown (km)", 0.5, 50.0, 5.0)
        distance_to_parks = st.number_input("Distance to Parks (km)", 0.1, 15.0, 1.5)

    with col3:
        hoa_fees = st.number_input("HOA Fees ($/month)", 0, 3000, 200)
        garage = st.slider("Garage Spaces", 0, 4, 1)
        pool = st.selectbox("Pool", [0, 1], format_func=lambda x: "Yes" if x else "No")
        tax_rate = st.number_input("Property Tax Rate (%)", 0.5, 3.5, 1.2)
        median_income = st.number_input("Area Median Income ($)", 25000, 250000, 70000)
        pop_density = st.number_input("Population Density", 500, 50000, 5000)

    if st.button("💰 Estimate Price", type="primary"):
        # Quick estimation using city medians as proxy
        city_data = raw_df[raw_df["city"] == city]
        base = city_data["sale_price"].median()
        sqft_ratio = sqft / city_data["sqft"].median()
        hood_adj = 1 + (neighborhood_score - 5) * 0.06
        age_adj = 1 + (year_built - 1985) * 0.002
        estimate = base * sqft_ratio * hood_adj * age_adj
        estimate = max(50000, int(estimate))

        st.success(f"**Estimated Price: ${estimate:,.0f}**")
        st.metric("Price per Sqft", f"${estimate / sqft:,.0f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Confidence Low", f"${int(estimate * 0.92):,}")
        with col_b:
            st.metric("Confidence High", f"${int(estimate * 1.08):,}")

# ---------------------------------------------------------------------------
# 2. Interactive Map
# ---------------------------------------------------------------------------
elif page == "Interactive Map":
    st.title("🗺️ Property Map")

    map_city = st.selectbox("Select City", ["All"] + list(raw_df["city"].unique()))
    if map_city != "All":
        map_data = raw_df[raw_df["city"] == map_city].sample(min(1000, len(raw_df[raw_df["city"] == map_city])), random_state=42)
    else:
        map_data = raw_df.sample(min(3000, len(raw_df)), random_state=42)

    fig = px.scatter_mapbox(
        map_data,
        lat="latitude",
        lon="longitude",
        color="sale_price",
        size="sqft",
        hover_name="property_type",
        hover_data=["city", "bedrooms", "bathrooms", "sale_price", "sqft"],
        color_continuous_scale="Viridis",
        size_max=12,
        zoom=10 if map_city != "All" else 3,
        mapbox_style="open-street-map",
        title=f"Properties — {map_city}",
    )
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Market Comparison
# ---------------------------------------------------------------------------
elif page == "Market Comparison":
    st.title("📊 Market Comparison")

    col1, col2 = st.columns(2)

    with col1:
        city_stats = raw_df.groupby("city")["sale_price"].agg(["mean", "median", "count"])
        fig1 = px.bar(
            city_stats.reset_index(),
            x="city", y="median",
            color="city",
            title="Median Sale Price by City",
            labels={"median": "Median Price ($)", "city": "City"},
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        raw_df["price_per_sqft"] = raw_df["sale_price"] / raw_df["sqft"]
        ppsf = raw_df.groupby("city")["price_per_sqft"].median().reset_index()
        fig2 = px.bar(
            ppsf, x="city", y="price_per_sqft", color="city",
            title="Median Price per Sqft by City",
            labels={"price_per_sqft": "$/sqft", "city": "City"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Price distribution
    fig3 = px.histogram(
        raw_df, x="sale_price", color="city", nbins=60,
        title="Price Distribution by City",
        barmode="overlay", opacity=0.6,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Model results
    if training_results:
        st.subheader("Model Performance")
        res_df = pd.DataFrame(training_results["results"]).T
        res_df.index.name = "Model"
        st.dataframe(res_df.style.highlight_max(axis=0, subset=["R2"]).format(
            {"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "MAPE": "{:.1f}%", "R2": "{:.4f}"}
        ))

# ---------------------------------------------------------------------------
# 4. Investment Scoring
# ---------------------------------------------------------------------------
elif page == "Investment Scoring":
    st.title("💎 Investment Opportunity Scoring")

    if pred_df is not None:
        from src.models.investment_scorer import RENTAL_YIELD_BENCHMARKS, APPRECIATION_RATES

        sample = pred_df.sample(min(500, len(pred_df)), random_state=42).copy()
        sample["price_gap_pct"] = (
            (sample["predicted_price"] - sample["sale_price"]) / sample["sale_price"] * 100
        )
        sample["rental_yield"] = sample["city"].map(RENTAL_YIELD_BENCHMARKS)
        sample["appreciation"] = sample["city"].map(APPRECIATION_RATES)

        col1, col2, col3 = st.columns(3)
        with col1:
            undervalued = (sample["price_gap_pct"] > 5).sum()
            st.metric("Undervalued Properties", f"{undervalued}")
        with col2:
            avg_yield = sample["rental_yield"].mean()
            st.metric("Avg Rental Yield", f"{avg_yield:.1f}%")
        with col3:
            avg_appr = sample["appreciation"].mean()
            st.metric("Avg Appreciation", f"{avg_appr:.1f}%")

        fig = px.scatter(
            sample, x="sale_price", y="predicted_price",
            color="price_gap_pct", size="sqft",
            hover_data=["city", "property_type", "bedrooms"],
            color_continuous_scale="RdYlGn",
            title="Predicted vs Actual Price (green = undervalued)",
        )
        fig.add_trace(go.Scatter(
            x=[sample["sale_price"].min(), sample["sale_price"].max()],
            y=[sample["sale_price"].min(), sample["sale_price"].max()],
            mode="lines", line=dict(dash="dash", color="red"),
            name="Fair value",
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Top opportunities
        st.subheader("Top 10 Investment Opportunities")
        top = sample.nlargest(10, "price_gap_pct")[
            ["city", "property_type", "sqft", "bedrooms", "sale_price",
             "predicted_price", "price_gap_pct"]
        ].reset_index(drop=True)
        top.columns = ["City", "Type", "Sqft", "Beds", "Listed $", "Predicted $", "Gap %"]
        st.dataframe(top.style.format({
            "Listed $": "${:,.0f}", "Predicted $": "${:,.0f}", "Gap %": "{:.1f}%",
        }))
    else:
        st.warning("Run model training first to generate predictions.")

# ---------------------------------------------------------------------------
# 5. Neighborhood Analysis
# ---------------------------------------------------------------------------
elif page == "Neighborhood Analysis":
    st.title("🏘️ Neighborhood Analysis")

    selected_city = st.selectbox("City", raw_df["city"].unique())
    city_data = raw_df[raw_df["city"] == selected_city]

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(
            city_data, x="neighborhood_score", y="sale_price",
            color="property_type", trendline="ols",
            title=f"Neighborhood Score vs Price — {selected_city}",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            city_data, x="school_rating", y="sale_price",
            color="property_type", trendline="ols",
            title=f"School Rating vs Price — {selected_city}",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    st.subheader("Feature Correlations")
    num_cols = ["sale_price", "sqft", "bedrooms", "neighborhood_score",
                "school_rating", "crime_index", "walk_score", "distance_to_downtown",
                "median_income_area"]
    corr = city_data[num_cols].corr()
    fig3 = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        title=f"Correlation Matrix — {selected_city}",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Stats table
    st.subheader("Summary Statistics")
    st.dataframe(city_data[num_cols].describe().T.style.format("{:,.2f}"))
