from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from src.decision_support import (
    calculate_scenarios,
    add_market_scenario_metrics,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mango FPO AI Decision Support System",
    page_icon="🥭",
    layout="wide",
)


# ============================================================
# FILES
# ============================================================

CURRENT_PREDICTIONS_FILE = Path(
    "data/processed/mango_current_predictions.csv"
)

CURRENT_FEATURES_FILE = Path(
    "data/processed/mango_current_features.csv"
)

CURRENT_DATA_FILE = Path(
    "data/processed/mango_combined_market_data.csv"
)

MODEL_FILE = Path(
    "models/mango_xgboost_model.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_predictions():
    df = pd.read_csv(CURRENT_PREDICTIONS_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date")


@st.cache_data
def load_current_features():
    df = pd.read_csv(CURRENT_FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data
def load_market_data():
    df = pd.read_csv(CURRENT_DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_FILE)
    return bundle["pipeline"]


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    CURRENT_PREDICTIONS_FILE,
    CURRENT_FEATURES_FILE,
    CURRENT_DATA_FILE,
    MODEL_FILE,
]

for file in required_files:
    if not file.exists():
        st.error(f"Required file not found: {file}")
        st.stop()


# ============================================================
# LOAD EVERYTHING
# ============================================================

predictions = load_predictions()
current_features = load_current_features()
market_data = load_market_data()
pipeline = load_model()

# Current market observations are used for the selection lists so the
# dashboard exposes all currently observed states, districts, markets,
# and Mango varieties. Prediction coverage can be narrower than the
# full current market-data coverage.
latest_market_data_date = market_data["date"].max()
current_data_start = latest_market_data_date - pd.Timedelta(days=30)
current_market_data = market_data[
    market_data["date"] >= current_data_start
].copy()


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_prediction_columns = {
    "date",
    "state",
    "district",
    "market",
    "variety",
    "modal_price",
    "predicted_next_price",
}

missing_prediction_columns = (
    required_prediction_columns - set(predictions.columns)
)

if missing_prediction_columns:
    st.error(
        "Prediction dataset is missing required columns: "
        + ", ".join(sorted(missing_prediction_columns))
    )
    st.stop()


# ============================================================
# DISPLAY HELPERS
# ============================================================

# These are display-only labels. The underlying dataset/model values
# are not changed.
VARIETY_DISPLAY_NAMES = {
    "Hapus(Alphaso)": "Hapus (Alphonso)",
    "Other": "Other / Unspecified",
}


def variety_label(value):
    return VARIETY_DISPLAY_NAMES.get(value, value)


# ============================================================
# HEADER
# ============================================================

st.title("🥭 Mango FPO AI Decision Support System")

st.caption(
    "AI/ML-based Mango market forecasting and decision-support "
    "prototype for Farmer Producer Organizations"
)


# ============================================================
# LATEST AVAILABLE DATA DATE
# ============================================================

latest_data_date = market_data["date"].max()

st.info(
    f"Latest available market dataset date: {latest_data_date.date()}. "
    "Selection lists use current market observations; model predictions "
    "are shown where prediction coverage is available."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("FPO Prediction Inputs")


# ------------------------------------------------------------
# MANGO VARIETY — ALL CURRENTLY OBSERVED VARIETIES
# ------------------------------------------------------------

varieties = sorted(
    current_market_data["variety"].dropna().unique()
)

if not varieties:
    st.error("No current Mango varieties are available in the dataset.")
    st.stop()

variety = st.sidebar.selectbox(
    f"Mango Variety ({len(varieties)} available)",
    varieties,
    format_func=variety_label,
)


# ------------------------------------------------------------
# STATE — ONLY STATES WHERE SELECTED VARIETY IS AVAILABLE
# ------------------------------------------------------------

variety_current_data = current_market_data[
    current_market_data["variety"] == variety
].copy()

states = sorted(
    variety_current_data["state"].dropna().unique()
)

if not states:
    st.error(
        f"No current states are available for {variety_label(variety)}."
    )
    st.stop()

state = st.sidebar.selectbox(
    f"State ({len(states)} available for selected variety)",
    states,
)


# ------------------------------------------------------------
# DISTRICT — ONLY DISTRICTS WHERE SELECTED VARIETY IS AVAILABLE
#                 IN THE SELECTED STATE
# ------------------------------------------------------------

state_current_data = variety_current_data[
    variety_current_data["state"] == state
].copy()

districts = sorted(
    state_current_data["district"].dropna().unique()
)

if not districts:
    st.error(
        f"No current districts are available for {variety_label(variety)} "
        f"in {state}."
    )
    st.stop()

district = st.sidebar.selectbox(
    f"District ({len(districts)} available)",
    districts,
)


# ------------------------------------------------------------
# MARKET — ONLY MARKETS WHERE SELECTED VARIETY IS AVAILABLE
#                  IN THE SELECTED STATE + DISTRICT
# ------------------------------------------------------------

district_current_data = state_current_data[
    state_current_data["district"] == district
].copy()

markets = sorted(
    district_current_data["market"].dropna().unique()
)

if not markets:
    st.error(
        f"No current markets are available for {variety_label(variety)}, "
        f"{district}, {state}."
    )
    st.stop()

market = st.sidebar.selectbox(
    f"Market ({len(markets)} available for selected location)",
    markets,
)


# ------------------------------------------------------------
# COVERAGE NOTE
# ------------------------------------------------------------

st.sidebar.caption(
    f"Availability follows the selected Mango variety: "
    f"{len(states)} states • {len(districts)} districts • "
    f"{len(markets)} markets"
)


# ------------------------------------------------------------
# QUANTITY
# ------------------------------------------------------------

quantity = st.sidebar.number_input(
    "Mango Quantity (kg)",
    min_value=100,
    max_value=100000,
    value=1000,
    step=100,
)


# ------------------------------------------------------------
# TRANSPORT COST
# ------------------------------------------------------------

transport_cost = st.sidebar.number_input(
    "Estimated Transport Cost (₹)",
    min_value=0.0,
    value=2500.0,
    step=500.0,
)


# ------------------------------------------------------------
# STORAGE COST
# ------------------------------------------------------------

storage_cost = st.sidebar.number_input(
    "Estimated Storage Cost (₹)",
    min_value=0.0,
    value=1000.0,
    step=250.0,
)


# ============================================================
# CURRENT MARKET RECORD
# ============================================================

selected_current = current_market_data[
    (current_market_data["state"] == state)
    & (current_market_data["district"] == district)
    & (current_market_data["market"] == market)
    & (current_market_data["variety"] == variety)
].copy()

if selected_current.empty:
    st.warning(
        f"Current market data does not contain a {variety_label(variety)} "
        f"observation for {market}, {district}, {state}. "
        "The location and variety lists show all currently observed values, "
        "but not every possible combination exists in the source data."
    )
    st.stop()

latest_current = selected_current.sort_values("date").iloc[-1]
current_price = float(latest_current["modal_price"])


# ============================================================
# MODEL PREDICTION FOR SELECTED COMBINATION
# ============================================================

selected = predictions[
    (predictions["state"] == state)
    & (predictions["district"] == district)
    & (predictions["market"] == market)
    & (predictions["variety"] == variety)
].copy()

prediction_available = not selected.empty

if prediction_available:
    latest = selected.sort_values("date").iloc[-1]
    predicted_price = float(latest["predicted_next_price"])
else:
    predicted_price = None


# ============================================================
# CURRENT + PREDICTED PRICE
# ============================================================

if prediction_available:
    # ========================================================
    # FPO SCENARIO CALCULATIONS
    # ========================================================

    scenarios = calculate_scenarios(
        predicted_price=predicted_price,
        current_price=current_price,
        quantity_kg=quantity,
        transport_cost=transport_cost,
        storage_cost=storage_cost,
    )

    # ========================================================
    # PRICE CHANGE
    # ========================================================

    price_difference = predicted_price - current_price

    if current_price != 0:
        percentage_change = (
            price_difference / current_price
        ) * 100
    else:
        percentage_change = 0

    # ========================================================
    # TOP METRICS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Latest Market Price",
        f"₹{current_price:,.0f}/quintal",
    )

    c2.metric(
        "Predicted Next Price",
        f"₹{predicted_price:,.0f}/quintal",
    )

    c3.metric(
        "Estimated Gross Revenue",
        f"₹{scenarios['predicted_revenue']:,.0f}",
    )

    c4.metric(
        "After Transport",
        f"₹{scenarios['predicted_after_transport']:,.0f}",
    )

    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================

    st.subheader("📊 Prediction Summary")

    if price_difference > 0:
        st.success(
            f"Predicted price change: +₹{price_difference:,.0f} "
            f"({percentage_change:.2f}%)"
        )

    elif price_difference < 0:
        st.warning(
            f"Predicted price change: ₹{price_difference:,.0f} "
            f"({percentage_change:.2f}%)"
        )

    else:
        st.info("The predicted price is approximately unchanged.")

    # ========================================================
    # FPO DECISION SUPPORT
    # ========================================================

    st.subheader("🤖 FPO Decision Support")

    st.write(
        "The system compares the latest observed market price with the "
        "model's predicted next observed price and calculates transparent "
        "revenue scenarios using the quantity, transport cost and storage "
        "cost entered by the FPO."
    )

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "Current-Price Revenue",
        f"₹{scenarios['current_revenue']:,.0f}",
    )

    d2.metric(
        "Predicted-Price Revenue",
        f"₹{scenarios['predicted_revenue']:,.0f}",
    )

    d3.metric(
        "After Transport + Storage",
        f"₹{scenarios['predicted_after_storage']:,.0f}",
    )

    # ========================================================
    # SCENARIO TABLE
    # ========================================================

    scenario_table = pd.DataFrame(
        {
            "Scenario": [
                "Sell at Latest Price",
                "Sell at Predicted Price",
                "Predicted Price - After Transport",
                "Predicted Price - After Transport + Storage",
            ],
            "Estimated Amount (₹)": [
                scenarios["current_revenue"],
                scenarios["predicted_revenue"],
                scenarios["predicted_after_transport"],
                scenarios["predicted_after_storage"],
            ],
        }
    )

    scenario_table["Estimated Amount (₹)"] = (
        scenario_table["Estimated Amount (₹)"].apply(
            lambda value: f"₹{value:,.0f}"
        )
    )

    st.dataframe(
        scenario_table,
        width="stretch",
        hide_index=True,
    )

else:
    # ========================================================
    # PREDICTION COVERAGE NOTICE
    # ========================================================

    st.subheader("📊 Prediction Summary")

    c1, c2 = st.columns(2)

    c1.metric(
        "Latest Market Price",
        f"₹{current_price:,.0f}/quintal",
    )

    c2.metric(
        "Predicted Next Price",
        "Not available",
    )

    st.info(
        "This market-variety combination is present in the current market "
        "data, but a model prediction is not currently available for this "
        "exact combination. The dashboard selection lists intentionally "
        "show the full current-data coverage."
    )

    st.caption(
        "Prediction coverage is narrower because the model requires the "
        "engineered historical/lag features used during training."
    )


# ============================================================
# HISTORICAL PRICE TREND
# ============================================================

st.subheader("📈 Historical Mango Price Trend")

trend = market_data[
    (market_data["state"] == state)
    & (market_data["district"] == district)
    & (market_data["market"] == market)
    & (market_data["variety"] == variety)
].sort_values("date")

if not trend.empty:
    fig = px.line(
        trend,
        x="date",
        y="modal_price",
        title=(
            f"{market} — {district}, {state} — "
            f"{variety_label(variety)}"
        ),
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Modal Price (₹/quintal)",
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

else:
    st.info(
        "No historical price records are available for the "
        "selected location and Mango variety."
    )


# ============================================================
# MARKET SCENARIO ANALYSIS
# ============================================================

st.subheader("🏪 Market Scenario Analysis")

st.write(
    f"Recent predicted market scenarios for "
    f"**{variety_label(variety)}** across **{state}**, "
    f"using the FPO-entered quantity, transport and storage assumptions."
)


# ------------------------------------------------------------
# RECENT 7-DAY WINDOW
# ------------------------------------------------------------

max_prediction_date = predictions["date"].max()
recent_start_date = max_prediction_date - pd.Timedelta(days=7)

recent = predictions[
    (predictions["state"] == state)
    & (predictions["variety"] == variety)
    & (predictions["date"] >= recent_start_date)
].copy()

if recent.empty:
    st.info(
        "No recent prediction records are available for this "
        "Mango variety in the selected state."
    )

else:
    # --------------------------------------------------------
    # LATEST OBSERVATION FOR EACH MARKET
    # --------------------------------------------------------

    market_latest = (
        recent
        .sort_values("date")
        .groupby(
            ["district", "market"],
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    # --------------------------------------------------------
    # ADD REVENUE SCENARIO METRICS
    # --------------------------------------------------------

    market_latest = add_market_scenario_metrics(
        market_latest,
        quantity_kg=quantity,
        transport_cost=transport_cost,
        storage_cost=storage_cost,
    )

    # --------------------------------------------------------
    # SELECTED MARKET FLAG
    # --------------------------------------------------------

    market_latest["Selected"] = (
        (market_latest["district"] == district)
        & (market_latest["market"] == market)
    ).map(
        {
            True: "✓",
            False: "",
        }
    )

    # --------------------------------------------------------
    # SORT BY ESTIMATED NET REVENUE
    # --------------------------------------------------------

    market_latest = market_latest.sort_values(
        "after_transport_storage",
        ascending=False,
    )

    # --------------------------------------------------------
    # MARKET DISPLAY LABEL
    # --------------------------------------------------------

    market_latest["Market Display"] = (
        market_latest["market"].astype(str)
        + " — "
        + market_latest["district"].astype(str)
    )

    # --------------------------------------------------------
    # PREPARE DISPLAY TABLE
    # --------------------------------------------------------

    display_table = market_latest[
        [
            "Selected",
            "date",
            "district",
            "market",
            "modal_price",
            "predicted_next_price",
            "predicted_change_percent",
            "gross_revenue",
            "after_transport",
            "after_transport_storage",
        ]
    ].copy()

    # --------------------------------------------------------
    # DATE FORMAT
    # --------------------------------------------------------

    display_table["date"] = (
        display_table["date"].dt.strftime("%d %b %Y")
    )

    # --------------------------------------------------------
    # NUMBER FORMAT
    # --------------------------------------------------------

    money_columns = [
        "modal_price",
        "predicted_next_price",
        "gross_revenue",
        "after_transport",
        "after_transport_storage",
    ]

    for col in money_columns:
        display_table[col] = (
            display_table[col]
            .round(0)
            .astype(int)
        )

    display_table["predicted_change_percent"] = (
        display_table["predicted_change_percent"].round(2)
    )

    # --------------------------------------------------------
    # RENAME COLUMNS
    # --------------------------------------------------------

    display_table.columns = [
        "Selected",
        "Latest Date",
        "District",
        "Market",
        "Current Price (₹/quintal)",
        "Predicted Next Price (₹/quintal)",
        "Predicted Change (%)",
        "Gross Revenue (₹)",
        "After Transport (₹)",
        "Estimated Net Revenue (₹)",
    ]

    # --------------------------------------------------------
    # SHOW TABLE
    # --------------------------------------------------------

    st.dataframe(
        display_table,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # SELECTED MARKET SCENARIO
    # --------------------------------------------------------

    selected_market_row = market_latest[
        (market_latest["district"] == district)
        & (market_latest["market"] == market)
    ]

    if not selected_market_row.empty:
        selected_market_row = selected_market_row.iloc[0]

        st.markdown(
            f"""
            ### 🎯 Selected Market: {market}

            **District:** {district}  
            **State:** {state}  
            **Current Price:** ₹{selected_market_row["modal_price"]:,.0f}/quintal  
            **Predicted Next Price:** ₹{selected_market_row["predicted_next_price"]:,.0f}/quintal  
            **Estimated Gross Revenue:** ₹{selected_market_row["gross_revenue"]:,.0f}  
            **After Transport:** ₹{selected_market_row["after_transport"]:,.0f}  
            **Estimated Net Revenue:** ₹{selected_market_row["after_transport_storage"]:,.0f}
            """
        )

    # --------------------------------------------------------
    # REVENUE COMPARISON CHART
    # --------------------------------------------------------

    st.subheader("💰 Estimated Net Revenue by Market")

    chart_data = market_latest[
        [
            "Market Display",
            "after_transport_storage",
        ]
    ].copy()

    chart_data = (
        chart_data
        .sort_values(
            "after_transport_storage",
            ascending=False,
        )
        .head(10)
    )

    chart_data = chart_data.rename(
        columns={
            "Market Display": "Market",
            "after_transport_storage":
                "Estimated Net Revenue (₹)",
        }
    )

    fig_revenue = px.bar(
        chart_data,
        x="Market",
        y="Estimated Net Revenue (₹)",
        title=(
            f"Top 10 Market Scenarios — "
            f"{variety_label(variety)} in {state}"
        ),
        text="Estimated Net Revenue (₹)",
    )

    fig_revenue.update_traces(
        texttemplate="₹%{text:,.0f}",
        textposition="outside",
    )

    fig_revenue.update_layout(
        xaxis_title="Market",
        yaxis_title="Estimated Net Revenue (₹)",
        xaxis_tickangle=-45,
    )

    st.plotly_chart(
        fig_revenue,
        width="stretch",
    )

    # --------------------------------------------------------
    # ASSUMPTION NOTE
    # --------------------------------------------------------

    st.caption(
        "Transport and storage values are FPO-entered scenario "
        "assumptions applied uniformly to the displayed markets. "
        "They are not observed market-specific logistics tariffs."
    )


# ============================================================
# DATA COVERAGE
# ============================================================

st.subheader("📚 Data Coverage")

coverage1, coverage2, coverage3, coverage4 = st.columns(4)

coverage1.metric(
    "Current Feature Records",
    f"{len(current_features):,}",
)

coverage2.metric(
    "Feature Markets",
    f"{current_features['market'].nunique():,}",
)

coverage3.metric(
    "Feature Varieties",
    f"{current_features['variety'].nunique():,}",
)

coverage4.metric(
    "Feature States",
    f"{current_features['state'].nunique():,}",
)

st.caption(
    f"Selection lists use the current market-data window ({current_data_start.date()} "
    f"to {latest_market_data_date.date()}) so the dashboard exposes the full "
    f"currently observed location and variety coverage. Model prediction "
    "coverage can be narrower because lag/rolling features are required."
)


# ============================================================
# METHODOLOGY & DATA NOTE
# ============================================================

with st.expander("ℹ️ Methodology & Data Note"):
    st.write(
        "Historical Mango market observations were used for model "
        "development and evaluation. Current 2026 market observations "
        "were processed through the engineered feature pipeline to "
        "generate model predictions."
    )

    st.write(
        "The XGBoost model predicts the next observed market-reporting "
        "price for the selected market and Mango variety when the required "
        "prediction features are available."
    )

    st.write(
        "Selection lists use current market observations and are filtered "
        "by the selected Mango variety, state and district so the "
        "displayed location choices represent observed combinations. "
        "Prediction outputs use the dedicated current prediction dataset, "
        "so some observed combinations may not yet have a model prediction."
    )

    st.write(
        "Transport and storage values are scenario inputs provided by "
        "the FPO user. They are not claimed to be observed "
        "market-specific logistics tariffs."
    )

    st.write(
        "Model outputs are predictive estimates and are not guaranteed "
        "market prices. Actual decisions should also consider market "
        "conditions, logistics, quality, spoilage and other operational "
        "factors."
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Mango FPO AI Decision Support System — Research Prototype"
)

st.markdown(
    """
    <div style="text-align: center; padding: 30px 0 10px 0; color: #777; font-size: 13px;">
        <strong>Made by Anuj Goyal</strong><br>
        Integrated M.Tech Artificial Intelligence | VIT Bhopal University
    </div>
    """,
    unsafe_allow_html=True,
)
