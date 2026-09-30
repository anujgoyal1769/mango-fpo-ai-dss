from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path("data/processed/mango_ml_ready.csv")
MODEL_FILE = Path("models/mango_price_model.joblib")
RESULT_FILE = Path("reports/model_results.csv")

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MANGO TOMORROW PRICE PREDICTION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.sort_values("date").reset_index(drop=True)

print("\nTotal records:", len(df))
print("Date range:", df["date"].min(), "to", df["date"].max())


# ============================================================
# TARGET
# ============================================================

TARGET = "next_day_modal_price"


# ============================================================
# FEATURES
# ============================================================

numeric_features = [
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
]

categorical_features = [
    "state",
    "district",
    "market",
    "variety",
]

all_features = numeric_features + categorical_features


# ============================================================
# REMOVE ROWS WITH MISSING MODEL VALUES
# ============================================================

model_df = df.dropna(
    subset=all_features + [TARGET]
).copy()

print("Records available for modelling:", len(model_df))


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

# 80% oldest observations -> training
# 20% newest observations -> testing

split_index = int(len(model_df) * 0.80)

train_df = model_df.iloc[:split_index].copy()
test_df = model_df.iloc[split_index:].copy()

print("\nTraining records:", len(train_df))
print("Testing records :", len(test_df))

print(
    "Training period :",
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)

print(
    "Testing period  :",
    test_df["date"].min(),
    "to",
    test_df["date"].max()
)


# ============================================================
# X / y
# ============================================================

X_train = train_df[all_features]
y_train = train_df[TARGET]

X_test = test_df[all_features]
y_test = test_df[TARGET]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

predictions = pipeline.predict(
    X_test
)


# ============================================================
# BASELINE
# ============================================================

# Simple baseline:
# tomorrow's price = today's modal price

baseline_predictions = X_test[
    "modal_price"
]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)


# ============================================================
# RANDOM FOREST METRICS
# ============================================================

rf_mae = mean_absolute_error(
    y_test,
    predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

rf_r2 = r2_score(
    y_test,
    predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print("\nBASELINE — Today's price as tomorrow's prediction")
print(f"MAE  : {baseline_mae:.2f}")
print(f"RMSE : {baseline_rmse:.2f}")
print(f"R²   : {baseline_r2:.4f}")

print("\nRANDOM FOREST")
print(f"MAE  : {rf_mae:.2f}")
print(f"RMSE : {rf_rmse:.2f}")
print(f"R²   : {rf_r2:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    [
        {
            "model": "Baseline - Today's Price",
            "MAE": baseline_mae,
            "RMSE": baseline_rmse,
            "R2": baseline_r2
        },
        {
            "model": "Random Forest",
            "MAE": rf_mae,
            "RMSE": rf_rmse,
            "R2": rf_r2
        }
    ]
)

results.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    {
        "pipeline": pipeline,
        "features": all_features,
        "target": TARGET
    },
    MODEL_FILE
)

print("\nSaved model:")
print(MODEL_FILE)

print("\nSaved results:")
print(RESULT_FILE)

print("\nDone.")