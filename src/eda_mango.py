from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# PATHS
# =========================================================

INPUT_FILE = Path("data/processed/mango_final.csv")
OUTPUT_DIR = Path("reports/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

for col in ["min_price", "max_price", "modal_price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("=" * 60)
print("MANGO EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("Records:", len(df))
print("Date range:", df["date"].min(), "to", df["date"].max())

# =========================================================
# 1. PRICE TREND
# =========================================================

daily_price = (
    df.groupby("date")["modal_price"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    daily_price["date"],
    daily_price["modal_price"]
)

plt.title("Average Mango Modal Price Over Time")
plt.xlabel("Date")
plt.ylabel("Average Modal Price (₹/quintal)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_price_trend.png",
    dpi=300
)

plt.close()

# =========================================================
# 2. MONTHLY PRICE PATTERN
# =========================================================

df["month"] = df["date"].dt.month

monthly_price = (
    df.groupby("month")["modal_price"]
    .mean()
)

plt.figure(figsize=(10, 6))

monthly_price.plot(
    kind="bar"
)

plt.title("Average Mango Modal Price by Month")
plt.xlabel("Month")
plt.ylabel("Average Modal Price (₹/quintal)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_monthly_price.png",
    dpi=300
)

plt.close()

# =========================================================
# 3. STATE-WISE PRICE
# =========================================================

state_price = (
    df.groupby("state")["modal_price"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(10, 7))

state_price.sort_values().plot(
    kind="barh"
)

plt.title("Top 15 States by Average Mango Modal Price")
plt.xlabel("Average Modal Price (₹/quintal)")
plt.ylabel("State")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_state_price.png",
    dpi=300
)

plt.close()

# =========================================================
# 4. MARKET-WISE PRICE
# =========================================================

market_price = (
    df.groupby("market")["modal_price"]
    .agg(["mean", "count"])
)

# Keep markets with at least 20 observations
market_price = market_price[
    market_price["count"] >= 20
]

market_price = (
    market_price["mean"]
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(10, 7))

market_price.sort_values().plot(
    kind="barh"
)

plt.title("Top 15 Markets by Average Mango Modal Price")
plt.xlabel("Average Modal Price (₹/quintal)")
plt.ylabel("Market")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_market_price.png",
    dpi=300
)

plt.close()

# =========================================================
# 5. VARIETY-WISE PRICE
# =========================================================

variety_price = (
    df.groupby("variety")["modal_price"]
    .agg(["mean", "count"])
)

variety_price = (
    variety_price[
        variety_price["count"] >= 20
    ]["mean"]
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 7))

variety_price.sort_values().plot(
    kind="barh"
)

plt.title("Average Mango Modal Price by Variety")
plt.xlabel("Average Modal Price (₹/quintal)")
plt.ylabel("Mango Variety")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_variety_price.png",
    dpi=300
)

plt.close()

# =========================================================
# 6. PRICE DISTRIBUTION
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["modal_price"],
    bins=40
)

plt.title("Distribution of Mango Modal Prices")
plt.xlabel("Modal Price (₹/quintal)")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_price_distribution.png",
    dpi=300
)

plt.close()

# =========================================================
# PRINT SUMMARY
# =========================================================

print("\nEDA completed successfully.")

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print("-", file)

print("\nTop states:")
print(state_price.to_string())

print("\nTop varieties:")
print(variety_price.head(15).to_string())

print("\nDone.")