from pathlib import Path
import pandas as pd


# ============================================================
# FILES
# ============================================================

MODEL_RESULTS_FILE = Path(
    "reports/model_results.csv"
)

MODEL_COMPARISON_FILE = Path(
    "reports/model_comparison.csv"
)

ABLATION_FILE = Path(
    "reports/feature_ablation_results.csv"
)

OUTPUT_FILE = Path(
    "reports/final_model_validation.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FINAL MODEL VALIDATION")
print("=" * 70)


# ============================================================
# LOAD AVAILABLE RESULTS
# ============================================================

frames = []

if MODEL_RESULTS_FILE.exists():

    df1 = pd.read_csv(
        MODEL_RESULTS_FILE
    )

    print("\nmodel_results.csv:")
    print(df1.to_string(index=False))

    frames.append(df1)


if MODEL_COMPARISON_FILE.exists():

    df2 = pd.read_csv(
        MODEL_COMPARISON_FILE
    )

    print("\nmodel_comparison.csv:")
    print(df2.to_string(index=False))

    frames.append(df2)


# ============================================================
# COMBINE
# ============================================================

if not frames:

    print(
        "\nNo existing model-results files were found."
    )

    raise SystemExit(1)


combined = pd.concat(
    frames,
    ignore_index=True
)

combined = combined.drop_duplicates()


# ============================================================
# STANDARD COLUMN NAMES
# ============================================================

rename_map = {
    "model": "Model",
    "Model": "Model",
    "name": "Model",
    "MAE": "MAE",
    "mae": "MAE",
    "RMSE": "RMSE",
    "rmse": "RMSE",
    "R2": "R2",
    "r2": "R2",
    "r_squared": "R2"
}

combined = combined.rename(
    columns={
        old: new
        for old, new in rename_map.items()
        if old in combined.columns
    }
)


# ============================================================
# KEEP USEFUL COLUMNS
# ============================================================

preferred_columns = [
    "Model",
    "MAE",
    "RMSE",
    "R2"
]

available_columns = [
    col
    for col in preferred_columns
    if col in combined.columns
]

final_results = combined[
    available_columns
].copy()


# ============================================================
# ROUND METRICS
# ============================================================

for col in ["MAE", "RMSE", "R2"]:

    if col in final_results.columns:

        final_results[col] = pd.to_numeric(
            final_results[col],
            errors="coerce"
        ).round(4)


# ============================================================
# REMOVE DUPLICATE MODELS
# ============================================================

if "Model" in final_results.columns:

    final_results = (
        final_results
        .drop_duplicates(
            subset=["Model"],
            keep="last"
        )
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    final_results.to_string(
        index=False
    )
)


print("\nSaved:")
print(OUTPUT_FILE)


# ============================================================
# ABLATION INFORMATION
# ============================================================

if ABLATION_FILE.exists():

    ablation = pd.read_csv(
        ABLATION_FILE
    )

    print("\n" + "=" * 70)
    print("FEATURE ABLATION RESULTS")
    print("=" * 70)

    print(
        ablation.to_string(
            index=False
        )
    )


print("\nDone.")