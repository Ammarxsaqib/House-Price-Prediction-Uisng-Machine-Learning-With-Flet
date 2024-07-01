import os
import joblib
import pandas as pd

# Get the current directory where the script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Path to the saved scaler and model relative to the current directory
scaler_path = os.path.join(current_directory, 'scaler.pkl')
model_path = os.path.join(current_directory, 'random_forest_model.pkl')

# Load the scaler and model
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# Load the new data from CSV relative to the current directory
new_data = pd.read_csv(os.path.join(current_directory, 'tested.csv'))

# Extract the feature columns (excluding the target column if present)
feature_columns = ['bathrooms', 'sqft_living', 'view', 'grade', 'sqft_above', 'sqft_living15']  # Adjust this list if necessary

# Select only the features from the new data
new_data_features = new_data[feature_columns]

# Ensure the new data matches the training data's feature names and order
new_data_scaled = scaler.transform(new_data_features)

# Make predictions
predictions = model.predict(new_data_scaled)
print(predictions)
