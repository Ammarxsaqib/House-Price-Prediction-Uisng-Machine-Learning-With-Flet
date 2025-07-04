House Price Prediction Project
📌 Overview
This project implements machine learning models to predict house prices based on various features like living area size, number of bathrooms, and construction quality. Three models were developed and compared: Linear Regression, Decision Trees, and Random Forests, with Random Forest emerging as the best performer.

🎯 Objectives
Analyze factors influencing house prices

Implement and compare multiple ML models

Develop an interactive prediction tool

Evaluate model performance using standard metrics

🛠️ Methodology
Data Preprocessing
Handled missing values (median imputation)

Applied Min-Max normalization

Feature engineering:

Price per square foot

Average living area of nearest neighbors (sqft_living15)

Models Implemented
Linear Regression (Baseline model)

Decision Trees (Captures non-linear relationships)

Random Forests (Ensemble method to reduce overfitting)

Evaluation Metrics
Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R-squared (R²) score

📊 Results
Best Performing Model: Random Forest

MAE: 149,281.33

MSE: 60,745,724,961.78

RMSE: 246,466.48

Key Influential Features:

Living area size (sqft_living)

Number of bathrooms

Construction quality (grade)

💻 Implementation
Web Interface: Developed using Flet for real-time predictions

Model Deployment: Best model (Random Forest) deployed for user interaction

🚀 How to Use
Clone the repository

Install requirements: pip install -r requirements.txt

Run the application: python app.py

Input house features in the web interface

View predicted price

📂 Project Structure
text
house-price-prediction/
├── data/                    # Dataset files
├── models/                  # Trained model files
├── notebooks/               # Jupyter notebooks for analysis
├── src/
│   ├── preprocessing.py     # Data cleaning and feature engineering
│   ├── train.py             # Model training scripts
│   └── app.py               # Web application
├── requirements.txt         # Python dependencies
└── README.md                # This file
🔧 Requirements
Python 3.8+

scikit-learn

pandas

numpy

Flet (for web interface)
