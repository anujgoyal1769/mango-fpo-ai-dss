from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path("data/processed/mango_ml_ready.csv")
MODEL_FILE = Path("models/mango_xgboost_model.joblib")

OUTPUT_DIR = Path("reports/xgboost_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

print("=" * 70)
print("XGBOOST ERROR ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

bundle = joblib.load(MODEL_FILE)

pipeline = bundle["pipeline"]
features = bundle["features"]
target = bundle["target"]


# ============================================================
# SAME TIME-BASED SPLIT
# ============================================================

df = df.sort_values("date").reset_index(drop=True)

model_df = df.dropna(
    subset=features + [target]
).copy()

split_index = int(len(model_df) * 0.80)

train_df = model_df.iloc[:split_index].copy()
test_df = model_df.iloc[split_index:].copy()

X_test = test_df[features]
y_test = test_df[target]

print("\nTest records:", len(test_df))


# ============================================================
# PREDICTION
# ============================================================

predictions = pipeline.predict(X_test)

test_df["predicted_price"] = predictions

test_df["absolute_error"] = (
    test_df[target] -
    test_df["predicted_price"]
).abs()

test_df["signed_error"] = (
    test_df["predicted_price"] -
    test_df[target]
)

test_df["percentage_error"] = (
    test_df["absolute_error"] /
    test_df[target].replace(0, np.nan)
) * 100


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_file = OUTPUT_DIR / "test_predictions.csv"

test_df.to_csv(
    prediction_file,
    index=False
)

print("\nSaved predictions:")
print(prediction_file)


# ============================================================
# ERROR SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

print(
    "Mean Absolute Error:",
    test_df["absolute_error"].mean()
)

print(
    "Median Absolute Error:",
    test_df["absolute_error"].median()
)

print(
    "Maximum Absolute Error:",
    test_df["absolute_error"].max()
)


# ============================================================
# LARGEST ERRORS
# ============================================================

print("\nLargest prediction errors:")

large_errors = (
    test_df[
        [
            "date",
            "state",
            "district",
            "market",
            "variety",
            target,
            "predicted_price",
            "absolute_error"
        ]
    ]
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(20)
)

print(
    large_errors.to_string(index=False)
)


# ============================================================
# ERROR BY VARIETY
# ============================================================

variety_error = (
    test_df
    .groupby("variety")
    .agg(
        records=(target, "count"),
        mae=("absolute_error", "mean"),
        mean_actual_price=(target, "mean"),
        mean_predicted_price=("predicted_price", "mean")
    )
    .sort_values("mae", ascending=False)
)

print("\nError by variety:")

print(
    variety_error.to_string()
)

variety_error.to_csv(
    OUTPUT_DIR / "error_by_variety.csv"
)


# ============================================================
# ERROR BY MONTH
# ============================================================

test_df["month"] = test_df["date"].dt.month

month_error = (
    test_df
    .groupby("month")
    .agg(
        records=(target, "count"),
        mae=("absolute_error", "mean"),
        mean_actual_price=(target, "mean")
    )
)

print("\nError by month:")

print(
    month_error.to_string()
)

month_error.to_csv(
    OUTPUT_DIR / "error_by_month.csv"
)


# ============================================================
# ERROR BY STATE
# ============================================================

state_error = (
    test_df
    .groupby("state")
    .agg(
        records=(target, "count"),
        mae=("absolute_error", "mean")
    )
    .sort_values("mae", ascending=False)
)

print("\nError by state:")

print(
    state_error.to_string()
)

state_error.to_csv(
    OUTPUT_DIR / "error_by_state.csv"
)


# ============================================================
# ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.5
)

min_value = min(
    y_test.min(),
    predictions.min()
)

max_value = max(
    y_test.max(),
    predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title(
    "Actual vs Predicted Mango Modal Price - XGBoost"
)

plt.xlabel(
    "Actual Next-Day Modal Price (₹/quintal)"
)

plt.ylabel(
    "Predicted Next-Day Modal Price (₹/quintal)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    test_df["signed_error"],
    bins=40
)

plt.axvline(
    0,
    linestyle="--"
)

plt.title(
    "XGBoost Prediction Error Distribution"
)

plt.xlabel(
    "Prediction Error (₹/quintal)"
)

plt.ylabel(
    "Number of Records"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

try:

    preprocessor = pipeline.named_steps["preprocessor"]

    model = pipeline.named_steps["model"]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .head(20)
    )

    print("\nTop 20 features:")

    print(
        importance.to_string(index=False)
    )

    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        importance["feature"][::-1],
        importance["importance"][::-1]
    )

    plt.title(
        "Top 20 XGBoost Feature Importances"
    )

    plt.xlabel(
        "Importance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_importance.png",
        dpi=300
    )

    plt.close()

except Exception as error:

    print(
        "\nFeature importance could not be generated:"
    )

    print(error)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST ANALYSIS COMPLETED")
print("=" * 70)

print("\nGenerated files:")
print("- test_predictions.csv")
print("- error_by_variety.csv")
print("- error_by_month.csv")
print("- error_by_state.csv")
print("- actual_vs_predicted.png")
print("- error_distribution.png")
print("- feature_importance.csv")
print("- feature_importance.png")