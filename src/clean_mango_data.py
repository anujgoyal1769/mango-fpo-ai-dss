from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

INPUT_FILE = Path("data/processed/mango_historical.csv")
OUTPUT_FILE = Path("data/processed/mango_clean.csv")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("=" * 60)
print("LOADING HISTORICAL MANGO DATA")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Original records:", len(df))

# ---------------------------------------------------------
# CLEAN COLUMN NAMES
# ---------------------------------------------------------

df.columns = df.columns.str.strip()

# ---------------------------------------------------------
# CONVERT DATE
# ---------------------------------------------------------

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# ---------------------------------------------------------
# CONVERT PRICE COLUMNS
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# REMOVE INVALID DATES
# ---------------------------------------------------------

before = len(df)

df = df.dropna(
    subset=["date"]
)

print("Invalid dates removed:", before - len(df))

# ---------------------------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------------------------

before = len(df)

df = df.drop_duplicates()

print("Duplicate rows removed:", before - len(df))

# ---------------------------------------------------------
# REMOVE INVALID PRICE RECORDS
# ---------------------------------------------------------

before = len(df)

df = df[
    (df["modal_price"] > 0) &
    (df["min_price"] >= 0) &
    (df["max_price"] >= 0)
].copy()

print("Invalid price rows removed:", before - len(df))

# ---------------------------------------------------------
# PRICE CONSISTENCY CHECK
# ---------------------------------------------------------

before = len(df)

df = df[
    (df["min_price"] <= df["modal_price"]) &
    (df["modal_price"] <= df["max_price"])
].copy()

print("Inconsistent price rows removed:", before - len(df))

# ---------------------------------------------------------
# ADD TIME FEATURES
# ---------------------------------------------------------

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.dayofyear
df["week"] = df["date"].dt.isocalendar().week.astype(int)

# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

df = df.sort_values(
    ["date", "state", "district", "market"]
)

# ---------------------------------------------------------
# SAVE CLEAN DATA
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ---------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CLEAN DATA SUMMARY")
print("=" * 60)

print("Final records :", len(df))
print(
    "Date range    :",
    df["date"].min(),
    "to",
    df["date"].max()
)

print(
    "States        :",
    df["state"].nunique()
)

print(
    "Districts     :",
    df["district"].nunique()
)

print(
    "Markets       :",
    df["market"].nunique()
)

print(
    "Varieties     :",
    df["variety"].nunique()
)

print("\nMissing values:")

print(
    df.isnull().sum().to_string()
)

print("\nPrice statistics:")

print(
    df[
        [
            "min_price",
            "max_price",
            "modal_price"
        ]
    ].describe().to_string()
)

print("\nSaved file:")
print(OUTPUT_FILE)
