from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/mango_clean.csv")
OUTPUT_FILE = Path("data/processed/mango_final.csv")

print("=" * 60)
print("FINALIZING MANGO DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Starting records:", len(df))

# ---------------------------------------------------------
# Treat zero minimum price as invalid/unreported
# ---------------------------------------------------------

zero_min = df["min_price"] == 0

print("Zero minimum-price rows:", zero_min.sum())

df = df[~zero_min].copy()

# ---------------------------------------------------------
# Final price consistency check
# ---------------------------------------------------------

invalid_price_order = (
    (df["min_price"] > df["modal_price"]) |
    (df["modal_price"] > df["max_price"])
)

print("Inconsistent price-order rows:", invalid_price_order.sum())

df = df[~invalid_price_order].copy()

# ---------------------------------------------------------
# Remove any remaining duplicates
# ---------------------------------------------------------

before = len(df)

df = df.drop_duplicates()

print("Additional duplicates removed:", before - len(df))

# ---------------------------------------------------------
# Sort data
# ---------------------------------------------------------

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "state", "district", "market"]
)

# ---------------------------------------------------------
# Save final dataset
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

print("Final records :", len(df))
print("Date range    :", df["date"].min(), "to", df["date"].max())
print("States        :", df["state"].nunique())
print("Districts     :", df["district"].nunique())
print("Markets       :", df["market"].nunique())
print("Varieties     :", df["variety"].nunique())

print("\nMissing values:")
print(df.isnull().sum().to_string())

print("\nPrice statistics:")
print(
    df[
        ["min_price", "max_price", "modal_price"]
    ].describe().to_string()
)

print("\nSaved file:")
print(OUTPUT_FILE)