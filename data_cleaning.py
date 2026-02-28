import pandas as pd

# Load workout dataset
df = pd.read_csv(r"Database/food.csv")

print("Original Shape:", df.shape)

# Remove unwanted columns
columns_to_remove = ['meal_id'] 
df = df.drop(columns=columns_to_remove, errors='ignore')

# Remove completely empty rows
df = df.dropna(how='all')

print("After Cleaning Shape:", df.shape)

# Save cleaned file
df.to_csv(r"Database/food.csv", index=False)

print("dataset cleaned successfully!")
