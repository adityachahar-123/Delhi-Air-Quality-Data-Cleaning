import pandas as pd
import numpy as np
import json
from pathlib import Path

INPUT_FILE = Path("data/raw_air_quality.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Initial profiling
profile_before = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum())
}

# Standardise column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Replace common missing-value markers
df = df.replace(["", "NA", "N/A", "null", "None", "-"], np.nan)

# Remove exact duplicate rows
df = df.drop_duplicates()

# Convert likely measurement columns to numeric
for col in df.columns:
    if any(word in col for word in ["pm", "aqi", "no2", "so2", "co", "o3"]):
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Flag negative pollutant readings as invalid
pollutant_cols = [
    col for col in df.columns
    if any(word in col for word in ["pm", "no2", "so2", "co", "o3"])
]

for col in pollutant_cols:
    df.loc[df[col] < 0, col] = np.nan

# Identify outliers using the IQR method
outlier_summary = {}

for col in df.select_dtypes(include="number").columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    if pd.notna(iqr) and iqr > 0:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_summary[col] = int(
            ((df[col] < lower) | (df[col] > upper)).sum()
        )

# Create scaled copies of numeric columns
for col in df.select_dtypes(include="number").columns:
    minimum = df[col].min()
    maximum = df[col].max()

    if pd.notna(minimum) and pd.notna(maximum) and maximum > minimum:
        df[col + "_scaled"] = (
            (df[col] - minimum) / (maximum - minimum)
        )

# Save cleaned dataset
df.to_csv(OUTPUT_DIR / "cleaned_data.csv", index=False)

# Save data-quality report
profile_after = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "outliers_detected": outlier_summary
}

with open(OUTPUT_DIR / "profile.json", "w") as file:
    json.dump(
        {"before_cleaning": profile_before,
         "after_cleaning": profile_after},
        file,
        indent=4
    )

print("Cleaning completed successfully.")
print("Results saved in the outputs folder.")
