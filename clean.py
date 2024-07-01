import pandas as pd

# Load the dataset
data = pd.read_csv("tested.csv")

# Print the initial data shape
print("Initial data shape:", data.shape)

# Check for missing values in the dataset
missing_data = data.isnull().sum()
print("Missing values in each column:\n", missing_data)

# Fill missing values with the mean of each column
data_filled = data.fillna(data.mean())

# Drop any unnamed columns
data_filled = data_filled.loc[:, ~data_filled.columns.str.contains('^Unnamed')]

# Print the filled data shape
print("Data shape after filling missing values:", data_filled.shape)

# Save the filled dataset
data_filled.to_csv("tested_filled.csv", index=False)

print("Dataset with filled missing values saved successfully.")
