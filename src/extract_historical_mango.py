from pathlib import Path
import pandas as pd

# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_FILE = Path(
    "data/raw/agricast_historical/"
    "agmarknet_india_historical_prices_2024_2025.csv"
)

OUTPUT_FILE = Path(
    "data/processed/mango_historical.csv"
)

print("=" * 60)
print("LOADING HISTORICAL AGMARKNET DATA")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nTotal records in original dataset:", len(df))

print("\nActual columns found:")
for col in df.columns:
    print("-", col)

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

# ============================================================
# FILTER MANGO
# ============================================================

print("\nSearching for Mango records...")

mango = df[
    df["Commodity"]
    .astype(str)
    .str.strip()
    .str.casefold()
    == "mango"
].copy()

print("Mango records found:", len(mango))

if mango.empty:
    raise ValueError("No Mango records were found.")

# ============================================================
# RENAME COLUMNS TO STANDARD NAMES
# ============================================================

mango = mango.rename(columns={
    "Sl no.": "sl_no",
    "District Name": "district",
    "Market Name": "market",
    "Commodity": "commodity",
    "Variety": "variety",
    "Grade": "grade",
    "Min Price (Rs./Quintal)": "min_price",
    "Max Price (Rs./Quintal)": "max_price",
    "Modal Price (Rs./Quintal)": "modal_price",
    "Price Date": "date",
    "State": "state"
})

# ============================================================
# CONVERT DATA TYPES
# ============================================================

mango["date"] = pd.to_datetime(
    mango["date"],
    errors="coerce"
)

for column in [
    "min_price",
    "max_price",
    "modal_price"
]:
    mango[column] = pd.to_numeric(
        mango[column],
        errors="coerce"
    )

# ============================================================
# REMOVE INVALID DATES
# ============================================================

before_invalid_dates = len(mango)

mango = mango.dropna(
    subset=["date"]
)

print(
    "Rows removed due to invalid dates:",
    before_invalid_dates - len(mango)
)

# ============================================================
# REMOVE DUPLICATES
# ============================================================

before_duplicates = len(mango)

mango = mango.drop_duplicates()

print(
    "Duplicate rows removed:",
    before_duplicates - len(mango)
)

# ============================================================
# SORT
# ============================================================

mango = mango.sort_values(
    ["date", "state", "district", "market"]
)

# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

mango.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("HISTORICAL MANGO DATA SUMMARY")
print("=" * 60)

print("Total Mango records :", len(mango))
print(
    "Date range          :",
    mango["date"].min(),
    "to",
    mango["date"].max()
)

print(
    "States              :",
    mango["state"].nunique()
)

print(
    "Districts           :",
    mango["district"].nunique()
)

print(
    "Markets             :",
    mango["market"].nunique()
)

print(
    "Varieties           :",
    mango["variety"].nunique()
)

print(
    "Grades              :",
    mango["grade"].nunique()
)

print(
    "Duplicate rows      :",
    mango.duplicated().sum()
)

# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    mango.isnull().sum().to_string()
)

# ============================================================
# TOP VARIETIES
# ============================================================

print("\nTop Mango varieties:")

print(
    mango["variety"]
    .value_counts()
    .head(20)
    .to_string()
)

# ============================================================
# PRICE SUMMARY
# ============================================================

print("\nPrice summary:")

print(
    mango[
        [
            "min_price",
            "max_price",
            "modal_price"
        ]
    ].describe().to_string()
)

print("\nSaved file:")

print(OUTPUT_FILE)