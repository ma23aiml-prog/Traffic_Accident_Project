
import os
import pandas as pd

# Input dataset
INPUT_FILE = "data/cleaned/accidents_cleaned.csv"

# Output folders and file
STAGING_DIR = "data/staging"
WEATHER_DIR = "data/raw/weather"
GRID_FILE = os.path.join(STAGING_DIR, "weather_grid_locations.csv")

os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(WEATHER_DIR, exist_ok=True)

# Load accident data
df = pd.read_csv(INPUT_FILE, low_memory=False)

# Keep valid dates and coordinates
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

df = df.dropna(subset=["date", "latitude", "longitude"])

# Group nearby coordinates into grid cells
df["grid_latitude"] = (df["latitude"] / 0.25).round() * 0.25
df["grid_longitude"] = (df["longitude"] / 0.25).round() * 0.25
# Count accident records in each grid cell
grid = (
    df.groupby(["grid_latitude", "grid_longitude"])
    .size()
    .reset_index(name="accident_count")
)

# Save the grid locations
grid.to_csv(GRID_FILE, index=False)

print("Weather grid preparation completed!")
print("Valid accident records:", len(df))
print("Unique weather grid cells:", len(grid))
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
print("Grid file saved to:", GRID_FILE)
print("\nSample grid locations:")
print(grid.head(10).to_string(index=False))