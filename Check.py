import pandas as pd

# Load the dataset
data = pd.read_csv("tested.csv")

# Print the initial data shape
print("Initial data shape:", data.shape)

# Check for missing values in the dataset
missing_data = data.isnull().sum()
total_missing = missing_data.sum()

# Print the missing values in each column
print("Missing values in each column:\n", missing_data)

# Summary of the dataset
if total_missing == 0:
    print("The dataset is OK for training. No missing values found.")
else:
    print(f"The dataset has {total_missing} missing values. It is not OK for training.")

    # Optionally, you can print the percentage of missing values for each column
    missing_percentage = (missing_data / len(data)) * 100
    print("Percentage of missing values in each column:\n", missing_percentage)
