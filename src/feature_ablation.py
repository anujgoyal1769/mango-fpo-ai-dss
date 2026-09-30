from pathlib import Path

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
OUTPUT_FILE = Path("reports/feature_ablation_results.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FEATURE ABLATION EXPERIMENT")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.sort_values("date").reset_index(drop=True)

TARGET = "next_day_modal_price"


# ============================================================
# FEATURE GROUPS
# ============================================================

# Experiment A:
# Today's modal price only
price_only = [
    "modal_price"
]

# Experiment B:
# Current + historical price information
historical_price = [
    "modal_price",
    "lag_1_price",
    "lag_3_price",
    "lag_7_price",
    "price_change_1",
    "price_change_3",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_std_7",
]

# Experiment C:
# Full available information
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

full_features = numeric_features + categorical_features


# ============================================================
# TIME-BASED SPLIT
# ============================================================

model_df = df.dropna(
    subset=[TARGET]
).copy()

split_index = int(len(model_df) * 0.80)

train_df = model_df.iloc[:split_index].copy()
test_df = model_df.iloc[split_index:].copy()

y_train = train_df[TARGET]
y_test = test_df[TARGET]

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
# FUNCTION TO TRAIN XGBOOST
# ============================================================

def train_xgb(
    train_data,
    test_data,
    feature_list
):
    X_train = train_data[feature_list]
    X_test = test_data[feature_list]

    numeric = [
        col for col in feature_list
        if col in numeric_features
    ]

    categorical = [
        col for col in feature_list
        if col in categorical_features
    ]

    transformers = []

    if numeric:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric
            )
        )

    if categorical:
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

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = XGBRegressor(
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

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

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

    return mae, rmse, r2


# ============================================================
# RUN EXPERIMENTS
# ============================================================

experiments = [
    (
        "A - Today's Modal Price",
        price_only
    ),
    (
        "B - Historical Price Features",
        historical_price
    ),
    (
        "C - Full Feature Set",
        full_features
    )
]

results = []

for name, feature_list in experiments:

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    valid_columns = feature_list + [TARGET]

    train_valid = train_df.dropna(
        subset=valid_columns
    ).copy()

    test_valid = test_df.dropna(
        subset=valid_columns
    ).copy()

    local_y_test = test_valid[TARGET]

    mae, rmse, r2 = train_xgb(
        train_valid,
        test_valid,
        feature_list
    )

    print("Features used:", len(feature_list))
    print("Test records :", len(test_valid))
    print(f"MAE          : {mae:.2f}")
    print(f"RMSE         : {rmse:.2f}")
    print(f"R²           : {r2:.4f}")

    results.append(
        {
            "experiment": name,
            "number_of_features": len(feature_list),
            "test_records": len(test_valid),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY FINAL TABLE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ABLATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nDone.")