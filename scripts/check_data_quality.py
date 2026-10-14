
import pandas as pd

file_path = "data/raw/accidents/dft-road-casualty-statistics-collision-2025.csv"
df = pd.read_csv(file_path, low_memory=False)

print("========== DATA QUALITY REPORT ==========")

# 1. Dataset size
print("\n1. Dataset shape:")
print(df.shape)

# 2. Duplicate accident records
print("\n2. Duplicate collision IDs:")
print(df["collision_index"].duplicated().sum())

# 3. Missing values
print("\n3. Columns with missing values:")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# 4. Accident severity distribution
print("\n4. Collision severity distribution:")
print(df["collision_severity"].value_counts(dropna=False))

# 5. Invalid coordinates
print("\n5. Invalid coordinates:")
invalid_coords = (
    df["latitude"].isna()
    | df["longitude"].isna()
    | ~df["latitude"].between(-90, 90)
    | ~df["longitude"].between(-180, 180)
)
print("Records with invalid coordinates:", invalid_coords.sum())

# 6. Number of vehicles and casualties
print("\n6. Vehicle count summary:")
print(df["number_of_vehicles"].describe())

print("\nCasualty count summary:")
print(df["number_of_casualties"].describe())

# 7. Date range
print("\n7. Accident date range:")
print(df["date"].min(), "to", df["date"].max())

print("\n========== REPORT COMPLETE ==========")