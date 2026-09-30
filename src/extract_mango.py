from pathlib import Path
import pandas as pd

# Folder containing all daily CSV files
DATA_DIR = Path("data/raw/mandi_archive/data")

# Output file
OUTPUT_FILE = Path("data/processed/mango_market.csv")

print("Searching for CSV files...")

csv_files = sorted(DATA_DIR.rglob("*.csv"))

print(f"Found {len(csv_files)} CSV files.")

mango_data = []

for file in csv_files:
    try:
        df = pd.read_csv(file)

        # Check whether commodity column exists
        if "commodity" not in df.columns:
            continue

        # Clean commodity names
        commodity = (
            df["commodity"]
            .astype(str)
            .str.strip()
            .str.casefold()
        )

        # Keep only Mango
        mango = df[commodity == "mango"].copy()

        if not mango.empty:
            mango_data.append(mango)
            print(f"{file.name} → {len(mango)} Mango records")

    except Exception as e:
        print(f"Error reading {file.name}: {e}")

# Check whether Mango was found
if not mango_data:
    raise ValueError("No Mango records were found.")

# Combine all Mango records
mango_df = pd.concat(mango_data, ignore_index=True)

# Convert date
mango_df["date"] = pd.to_datetime(
    mango_df["date"],
    errors="coerce"
)

# Convert prices to numbers
for column in ["min_price", "max_price", "modal_price"]:
    mango_df[column] = pd.to_numeric(
        mango_df[column],
        errors="coerce"
    )

# Remove exact duplicate rows
mango_df = mango_df.drop_duplicates()

# Sort by date
mango_df = mango_df.sort_values("date")

# Create processed folder
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

# Save Mango dataset
mango_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==============================")
print("MANGO DATA SUMMARY")
print("==============================")

print("Total Mango records :", len(mango_df))
print("Date range           :", mango_df["date"].min(), "to", mango_df["date"].max())
print("States               :", mango_df["state"].nunique())
print("Districts            :", mango_df["district"].nunique())
print("Markets              :", mango_df["market"].nunique())
print("Varieties            :", mango_df["variety"].nunique())

print("\nMissing values:")
print(mango_df.isnull().sum())

print("\nVarieties:")
print(mango_df["variety"].value_counts().head(20))

print("\nSaved file:")
print(OUTPUT_FILE)