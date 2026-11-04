import requests
from pathlib import Path

# Add the direct public CSV download link here
DATA_URL = "PASTE_YOUR_DATASET_CSV_LINK_HERE"

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "raw_air_quality.csv"

if "PASTE_YOUR" in DATA_URL:
    raise ValueError("Please add your dataset CSV link in DATA_URL.")

response = requests.get(DATA_URL, timeout=60)
response.raise_for_status()

if "text/csv" not in response.headers.get("Content-Type", "").lower():
    print("Warning: The server did not identify the response as CSV.")

OUTPUT_FILE.write_bytes(response.content)

print(f"Dataset saved to: {OUTPUT_FILE}")
