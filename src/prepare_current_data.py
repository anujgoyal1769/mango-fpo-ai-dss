from pathlib import Path
import pandas as pd

HISTORICAL_FILE = Path(
    "data/processed/mango_final.csv"
)

CURRENT_FILE = Path(
    "data/processed/mango_market.csv"
)

OUTPUT_FILE = Path(
    "data/processed/mango_combined_market_data.csv"
)

print("=" * 70)
print("PREPARING HISTORICAL + CURRENT MANGO DATA")
print("=" * 70)

# ---------------------------------------------------------
# LOAD HISTORICAL DATA
# ---------------------------------------------------------

historical = pd.read_csv(HISTORICAL_FILE)

historical["date"] = pd.to_datetime(
    historical["date"],
    errors="coerce"
)

print("\nHistorical records:", len(historical))

# ---------------------------------------------------------
# LOAD CURRENT DATA
# ---------------------------------------------------------

current = pd.read_csv(CURRENT_FILE)

current["date"] = pd.to_datetime(
    current["date"],
    errors="coerce"
)

print("Current records:", len(current))

# ---------------------------------------------------------
# KEEP COMMON COLUMNS
# ---------------------------------------------------------

common_columns = [
    "date",
    "state",
    "district",
    "market",
    "commodity",
    "variety",
    "grade",
    "min_price",
    "max_price",
    "modal_price"
]

historical = historical[common_columns].copy()
current = current[common_columns].copy()

# ---------------------------------------------------------
# COMBINE
# ---------------------------------------------------------

combined = pd.concat(
    [historical, current],
    ignore_index=True
)

# ---------------------------------------------------------
# CLEAN
# ---------------------------------------------------------

combined["min_price"] = pd.to_numeric(
    combined["min_price"],
    errors="coerce"
)

combined["max_price"] = pd.to_numeric(
    combined["max_price"],
    errors="coerce"
)

combined["modal_price"] = pd.to_numeric(
    combined["modal_price"],
    errors="coerce"
)

combined = combined.dropna(
    subset=[
        "date",
        "state",
        "district",
        "market",
        "variety",
        "modal_price"
    ]
)

# ---------------------------------------------------------
# REMOVE INVALID PRICES
# ---------------------------------------------------------

combined = combined[
    combined["modal_price"] > 0
].copy()

# ---------------------------------------------------------
# REMOVE EXACT DUPLICATES
# ---------------------------------------------------------

combined = combined.drop_duplicates()

# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

combined = combined.sort_values(
    [
        "date",
        "state",
        "district",
        "market",
        "variety"
    ]
).reset_index(drop=True)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

combined.to_csv(
    OUTPUT_FILE,
    index=False
)

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("COMBINED DATA SUMMARY")
print("=" * 70)

print(
    "Total records:",
    len(combined)
)

print(
    "Date range:",
    combined["date"].min(),
    "to",
    combined["date"].max()
)

print(
    "Historical records:",
    (combined["date"] < "2026-01-01").sum()
)

print(
    "2026 records:",
    (combined["date"] >= "2026-01-01").sum()
)

print(
    "States:",
    combined["state"].nunique()
)

print(
    "Markets:",
    combined["market"].nunique()
)

print(
    "Varieties:",
    combined["variety"].nunique()
)

print("\nLatest 2026 date:")

print(
    combined[
        combined["date"] >= "2026-01-01"
    ]["date"].max()
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nDone.")