from pathlib import Path
import pandas as pd

# =========================================================
# FILE PATHS
# =========================================================

INPUT_FILE = Path("data/processed/mango_final.csv")
OUTPUT_FILE = Path("data/processed/mango_ml_ready.csv")

# =========================================================
# LOAD DATA
# =========================================================

print("=" * 70)
print("MANGO PRICE PREDICTION - FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal records:", len(df))

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = df.columns.str.strip()

# =========================================================
# CONVERT DATA TYPES
# =========================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

price_columns = [
    "min_price",
    "max_price",
    "modal_price"
]

for col in price_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# =========================================================
# REMOVE INVALID ROWS
# =========================================================

df = df.dropna(
    subset=[
        "date",
        "state",
        "district",
        "market",
        "variety",
        "min_price",
        "max_price",
        "modal_price"
    ]
).copy()

# =========================================================
# DAILY MARKET-VARIETY AGGREGATION
# =========================================================
#
# If multiple grades exist for the same market/variety/day,
# calculate the daily mean of the reported price fields.
#
# This creates one observation per:
# date + state + district + market + variety
#
# =========================================================

group_columns = [
    "date",
    "state",
    "district",
    "market",
    "variety"
]

daily = (
    df.groupby(group_columns, as_index=False)
      .agg(
          min_price=("min_price", "mean"),
          max_price=("max_price", "mean"),
          modal_price=("modal_price", "mean")
      )
)

print(
    "\nDaily market-variety records after aggregation:",
    len(daily)
)

# =========================================================
# SORT
# =========================================================

daily = daily.sort_values(
    [
        "state",
        "district",
        "market",
        "variety",
        "date"
    ]
).reset_index(drop=True)

# =========================================================
# DATE FEATURES
# =========================================================

daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month
daily["day"] = daily["date"].dt.day
daily["day_of_year"] = daily["date"].dt.dayofyear
daily["day_of_week"] = daily["date"].dt.dayofweek
daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)

# =========================================================
# LAG FEATURES
# =========================================================
#
# shift(1) = previous observed price
# shift(3) = price from three previous observations
# shift(7) = price from seven previous observations
#
# These use only PAST information.
# =========================================================

group = daily.groupby(
    ["state", "district", "market", "variety"],
    group_keys=False
)

daily["lag_1_price"] = group["modal_price"].shift(1)
daily["lag_3_price"] = group["modal_price"].shift(3)
daily["lag_7_price"] = group["modal_price"].shift(7)

# =========================================================
# PRICE CHANGE FEATURES
# =========================================================

daily["price_change_1"] = (
    daily["modal_price"]
    - daily["lag_1_price"]
)

daily["price_change_3"] = (
    daily["modal_price"]
    - daily["lag_3_price"]
)

# =========================================================
# ROLLING PRICE FEATURES
# =========================================================
#
# shift(1) is used BEFORE rolling so today's price itself
# does not leak into the rolling historical average.
# =========================================================

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

# =========================================================
# CREATE TOMORROW'S DATE
# =========================================================

daily["target_date"] = (
    daily["date"]
    + pd.Timedelta(days=1)
)

# =========================================================
# CREATE TOMORROW'S TARGET PRICE
# =========================================================
#
# We want:
#
# Today's information
#          ↓
# Tomorrow's modal price
#
# Only the SAME market + variety is matched.
# =========================================================

target_columns = [
    "state",
    "district",
    "market",
    "variety",
    "date",
    "modal_price"
]

target_df = daily[target_columns].copy()

target_df = target_df.rename(
    columns={
        "date": "target_date",
        "modal_price": "next_day_modal_price"
    }
)

daily = daily.merge(
    target_df,
    on=[
        "state",
        "district",
        "market",
        "variety",
        "target_date"
    ],
    how="left"
)

# =========================================================
# CHECK TARGET AVAILABILITY
# =========================================================

target_available = daily[
    "next_day_modal_price"
].notna()

print(
    "\nRows with an actual next-day price:",
    target_available.sum()
)

print(
    "Rows without next-day price:",
    (~target_available).sum()
)

# =========================================================
# KEEP ONLY ROWS THAT CAN BE USED FOR TOMORROW PREDICTION
# =========================================================

ml_df = daily[
    target_available
].copy()

# Require enough history for lag/rolling features
required_features = [
    "lag_1_price",
    "lag_3_price",
    "lag_7_price",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_std_7"
]

ml_df = ml_df.dropna(
    subset=required_features
).copy()

# =========================================================
# SAVE
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

ml_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("ML-READY DATASET SUMMARY")
print("=" * 70)

print("Final ML records :", len(ml_df))
print(
    "Date range       :",
    ml_df["date"].min(),
    "to",
    ml_df["date"].max()
)

print(
    "Markets          :",
    ml_df["market"].nunique()
)

print(
    "Varieties        :",
    ml_df["variety"].nunique()
)

print(
    "States           :",
    ml_df["state"].nunique()
)

print("\nML Features:")

features = [
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
    "week_of_year"
]

for feature in features:
    print("-", feature)

print("\nTarget:")
print("- next_day_modal_price")

print("\nSaved file:")
print(OUTPUT_FILE)

print("\nDone.")