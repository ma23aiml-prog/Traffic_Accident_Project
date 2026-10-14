
import pandas as pd

file_path = "data/raw/accidents/dft-road-casualty-statistics-collision-2025.csv"

df = pd.read_csv(file_path, low_memory=False)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values in the top 15 columns:")
print(df.isnull().sum().sort_values(ascending=False).head(15))