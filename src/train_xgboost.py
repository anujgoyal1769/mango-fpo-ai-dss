from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path("data/processed/mango_ml_ready.csv")
MODEL_FILE = Path("models/mango_xgboost_model.joblib")
RESULT_FILE = Path("reports/model_comparison.csv")

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MANGO TOMORROW PRICE PREDICTION - XGBOOST")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.sort_values("date").reset_index(drop=True)

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
# REMOVE MISSING VALUES FOR MODEL INPUT
# ============================================================

model_df = df.dropna(
    subset=all_features + [TARGET]
).copy()

print("Records:", len(model_df))


# ============================================================
# TIME-BASED SPLIT
# ============================================================

split_index = int(len(model_df) * 0.80)

train_df = model_df.iloc[:split_index].copy()
test_df = model_df.iloc[split_index:].copy()

print("\nTraining records:", len(train_df))
print("Testing records :", len(test_df))

print(
    "Training period:",
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)

print(
    "Testing period :",
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
            OneHotEncoder(handle_unknown="ignore")
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
# XGBOOST MODEL
# ============================================================

xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="mae",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining XGBoost...")

pipeline.fit(
    X_train,
    y_train
)

print("XGBoost training completed.")


# ============================================================
# PREDICTION
# ============================================================

predictions = pipeline.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


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


# ============================================================
# SAVE XGBOOST RESULT
# ============================================================

result = pd.DataFrame(
    [{
        "model": "XGBoost",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }]
)

result.to_csv(
    RESULT_FILE,
    index=False
)

print("\nSaved model:")
print(MODEL_FILE)

print("\nSaved result:")
print(RESULT_FILE)

print("\nDone.")