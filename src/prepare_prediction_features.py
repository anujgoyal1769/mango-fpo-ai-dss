from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "data/processed/mango_combined_market_data.csv"
)

OUTPUT_FILE = Path(
    "data/processed/mango_current_features.csv"
)

print("=" * 70)
print("PREPARING CURRENT MANGO PREDICTION FEATURES")
print("=" * 70)

# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

for col in ["min_price", "max_price", "modal_price"]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=[
        "date",
        "state",
        "district",
        "market",
        "variety",
        "modal_price"
    ]
).copy()

# ---------------------------------------------------------
# REMOVE INVALID PRICES
# ---------------------------------------------------------

df = df[
    df["modal_price"] > 0
].copy()

# ---------------------------------------------------------
# DAILY MARKET-VARIETY AGGREGATION
# ---------------------------------------------------------

group_cols = [
    "date",
    "state",
    "district",
    "market",
    "variety"
]

daily = (
    df.groupby(
        group_cols,
        as_index=False
    )
    .agg(
        min_price=("min_price", "mean"),
        max_price=("max_price", "mean"),
        modal_price=("modal_price", "mean")
    )
)

# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

daily = daily.sort_values(
    [
        "state",
        "district",
        "market",
        "variety",
        "date"
    ]
).reset_index(drop=True)

group = daily.groupby(
    [
        "state",
        "district",
        "market",
        "variety"
    ],
    group_keys=False
)

# ---------------------------------------------------------
# DATE FEATURES
# ---------------------------------------------------------

daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month
daily["day"] = daily["date"].dt.day
daily["day_of_year"] = daily["date"].dt.dayofyear
daily["day_of_week"] = daily["date"].dt.dayofweek
daily["week_of_year"] = (
    daily["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

# ---------------------------------------------------------
# LAG FEATURES
# ---------------------------------------------------------

daily["lag_1_price"] = (
    group["modal_price"].shift(1)
)

daily["lag_3_price"] = (
    group["modal_price"].shift(3)
)

daily["lag_7_price"] = (
    group["modal_price"].shift(7)
)

# ---------------------------------------------------------
# PRICE CHANGE
# ---------------------------------------------------------

daily["price_change_1"] = (
    daily["modal_price"]
    - daily["lag_1_price"]
)

daily["price_change_3"] = (
    daily["modal_price"]
    - daily["lag_3_price"]
)

# ---------------------------------------------------------
# ROLLING FEATURES
# ONLY PAST OBSERVATIONS
# ---------------------------------------------------------

daily["rolling_mean_3"] = (
    group["modal_price"]
    .transform(
        lambda x:
        x.shift(1).rolling(3).mean()
    )
)

daily["rolling_mean_7"] = (
    group["modal_price"]
    .transform(
        lambda x:
        x.shift(1).rolling(7).mean()
    )
)

daily["rolling_std_7"] = (
    group["modal_price"]
    .transform(
        lambda x:
        x.shift(1).rolling(7).std()
    )
)

# ---------------------------------------------------------
# KEEP ONLY 2026 DATA
# ---------------------------------------------------------

current_features = daily[
    daily["date"] >= "2026-01-01"
].copy()

# ---------------------------------------------------------
# KEEP ROWS WITH ALL MODEL FEATURES
# ---------------------------------------------------------

required = [
    "min_price",
    "max_price",
    "modal_price",
    "lag_1_price",
    "lag_3_price",
    "lag_7_price",
    "price_change_1",
    "price_change_3",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_std_7",
    "year",
    "month",
    "day",
    "day_of_year",
    "day_of_week",
    "week_of_year",
    "state",
    "district",
    "market",
    "variety"
]

current_features = current_features.dropna(
    subset=required
).copy()

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

current_features.to_csv(
    OUTPUT_FILE,
    index=False
)

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CURRENT PREDICTION FEATURE SUMMARY")
print("=" * 70)

print(
    "Rows with usable features:",
    len(current_features)
)

print(
    "Date range:",
    current_features["date"].min(),
    "to",
    current_features["date"].max()
)

print(
    "Markets:",
    current_features["market"].nunique()
)

print(
    "Varieties:",
    current_features["variety"].nunique()
)

print(
    "States:",
    current_features["state"].nunique()
)

print("\nLatest feature date:")

print(
    current_features["date"].max()
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nDone.")    