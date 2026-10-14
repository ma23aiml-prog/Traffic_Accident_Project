
import pandas as pd
from pathlib import Path

SOURCE = Path("data/raw/accidents/dft-road-casualty-statistics-collision-2025.csv")
CLEAN_DIR = Path("data/cleaned")
REJECT_DIR = Path("data/rejected")

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
REJECT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE, low_memory=False)

# Normalize column names and prepare validation
df.columns = df.columns.str.strip()

# Reject records missing essential identifiers or severity
missing_required = (
    df["collision_index"].isna()
    | df["collision_severity"].isna()
)

# Reject invalid or missing coordinates for location-based analysis
invalid_coordinates = (
    df["latitude"].isna()
    | df["longitude"].isna()
    | ~df["latitude"].between(-90, 90)
    | ~df["longitude"].between(-180, 180)
)

# Reject invalid vehicle/casualty counts
invalid_counts = (
    (df["number_of_vehicles"] < 1)
    | (df["number_of_casualties"] < 1)
)

# Record all applicable rejection reasons
reasons = pd.Series("", index=df.index, dtype="object")

checks = [
    (missing_required, "missing_required_identifier_or_severity"),
    (invalid_coordinates, "missing_or_invalid_coordinates"),
    (invalid_counts, "invalid_vehicle_or_casualty_count"),
]

for mask, reason in checks:
    reasons.loc[mask] = reasons.loc[mask].apply(
        lambda existing: f"{existing};{reason}" if existing else reason
    )

rejected_mask = reasons.ne("")

rejected = df.loc[rejected_mask].copy()
rejected["rejection_reason"] = reasons.loc[rejected_mask]

cleaned = df.loc[~rejected_mask].copy()

# Parse dates using the source's day/month/year format
cleaned["date"] = pd.to_datetime(
    cleaned["date"], format="%d/%m/%Y", errors="coerce"
)

# Keep date parsing failures out of the cleaned dataset
bad_dates = cleaned["date"].isna()
if bad_dates.any():
    bad_date_rows = cleaned.loc[bad_dates].copy()
    bad_date_rows["rejection_reason"] = "invalid_date"
    rejected = pd.concat([rejected, bad_date_rows], ignore_index=True)
    cleaned = cleaned.loc[~bad_dates].copy()

cleaned.to_csv(CLEAN_DIR / "accidents_cleaned.csv", index=False)
rejected.to_csv(REJECT_DIR / "accidents_rejected.csv", index=False)

print("Cleaning complete!")
print("Source records:", len(df))
print("Cleaned records:", len(cleaned))
print("Rejected records:", len(rejected))
print("\nRejection reasons:")
if len(rejected):
    print(rejected["rejection_reason"].value_counts())
else:
    print("No rejected records")
print("\nCleaned file:", CLEAN_DIR / "accidents_cleaned.csv")
print("Rejected file:", REJECT_DIR / "accidents_rejected.csv")