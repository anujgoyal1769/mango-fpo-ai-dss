from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    "data/processed/mango_current_features.csv"
)

MODEL_FILE = Path(
    "models/mango_xgboost_model.joblib"
)

OUTPUT_FILE = Path(
    "data/processed/mango_current_predictions.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CURRENT MANGO PRICE PREDICTIONS")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# ============================================================
# LOAD MODEL
# ============================================================

bundle = joblib.load(
    MODEL_FILE
)

pipeline = bundle["pipeline"]
features = bundle["features"]


print("\nCurrent feature records:", len(df))
print(
    "Date range:",
    df["date"].min(),
    "to",
    df["date"].max()
)


# ============================================================
# KEEP ONLY ROWS WITH REQUIRED FEATURES
# ============================================================

prediction_df = df.dropna(
    subset=features
).copy()


# ============================================================
# PREDICT
# ============================================================

X = prediction_df[features]

prediction_df["predicted_next_price"] = (
    pipeline.predict(X)
)


# ============================================================
# CALCULATE PRICE CHANGE
# ============================================================

prediction_df["predicted_change"] = (
    prediction_df["predicted_next_price"]
    - prediction_df["modal_price"]
)

prediction_df["predicted_change_percent"] = (
    prediction_df["predicted_change"]
    / prediction_df["modal_price"]
    * 100
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)

print(
    "Predictions generated:",
    len(prediction_df)
)

print(
    "Markets:",
    prediction_df["market"].nunique()
)

print(
    "Varieties:",
    prediction_df["variety"].nunique()
)

print(
    "States:",
    prediction_df["state"].nunique()
)

# ============================================================
# LATEST OBSERVATIONS
# ============================================================

latest_date = prediction_df["date"].max()

latest = prediction_df[
    prediction_df["date"] == latest_date
].copy()

print(
    "\nLatest prediction date:",
    latest_date
)

print(
    "Number of latest predictions:",
    len(latest)
)

# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\nSample latest predictions:")

columns_to_show = [
    "date",
    "state",
    "district",
    "market",
    "variety",
    "modal_price",
    "predicted_next_price",
    "predicted_change",
    "predicted_change_percent"
]

print(
    latest[
        columns_to_show
    ]
    .sort_values(
        "market"
    )
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE LATEST PREDICTIONS
# ============================================================

latest_file = Path(
    "data/processed/mango_latest_predictions.csv"
)

latest.to_csv(
    latest_file,
    index=False
)

print("\nSaved complete predictions:")
print(OUTPUT_FILE)

print("\nSaved latest predictions:")
print(latest_file)

print("\nDone.")