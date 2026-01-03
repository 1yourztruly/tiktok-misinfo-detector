import pandas as pd

# Load the dataset
df = pd.read_csv("data/tiktok_scroll.csv")

# Basic info
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 3 rows:")
print(df.head(3).to_string(index=False))

