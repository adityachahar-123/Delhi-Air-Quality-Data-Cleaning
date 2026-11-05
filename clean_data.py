
import pandas as pd
import numpy as np
import json
from pathlib import Path

# File locations
INPUT_FILE = Path("data/air_quality_historical.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load original dataset
df = pd.read_csv(INPUT_FILE)

# Standardise column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Initial data profiling
before = {
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum())
}

# Convert date column
if "date" in df.columns:
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

# Convert other columns to numeric where possible
for col in df.columns:
    if col != "date":
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted

# Remove duplicate records
df = df.drop_duplicates()

# Handle invalid negative pollutant readings
pollutant_cols = [
    col for col in df.columns
    if any(x in col for x in ["pm", "no2", "so2", "co", "o3"])
]

for col in pollutant_cols:
    df.loc[df[col] < 0, col] = np.nan

# Missing-value treatment
numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    median_value = df[col].median()
    if pd.notna(median_value):
        df[col] = df[col].fillna(median_value)

# Outlier detection using IQR
outliers = {}

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    if pd.notna(iqr) and iqr > 0:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers[col] = int(
            ((df[col] < lower) | (df[col] > upper)).sum()
        )

# Min-Max normalization
for col in numeric_cols:
    minimum = df[col].min()
    maximum = df[col].max()

    if pd.notna(minimum) and maximum > minimum:
        df[col + "_scaled"] = (
            (df[col] - minimum) / (maximum - minimum)
        )

# Final profiling
after = {
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum())
}

# Save cleaned dataset
df.to_csv(OUTPUT_DIR / "cleaned_data.csv", index=False)

# Save profiling report
report = {
    "before_cleaning": before,
    "after_cleaning": after,
    "outliers_detected": outliers,
    "cleaning_methods": [
        "Column name standardization",
        "Date conversion",
        "Duplicate removal",
        "Invalid negative value handling",
        "Median imputation",
        "IQR outlier detection",
        "Min-Max normalization"
    ]
}

with open(OUTPUT_DIR / "profile.json", "w") as f:
    json.dump(report, f, indent=4, default=str)

print("Data cleaning completed!")
print("Cleaned dataset saved in outputs/cleaned_data.csv")
print("Profile saved in outputs/profile.json")
print("\nBefore cleaning:", before)
print("After cleaning:", after)
