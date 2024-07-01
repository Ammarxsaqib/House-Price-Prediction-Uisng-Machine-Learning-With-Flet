import flet as ft
import joblib
import numpy as np

# Load the trained model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("random_forest_model.pkl")

def predict_price(e):
    try:
        # Get input values
        bathrooms = float(bathrooms_input.value)
        sqft_living = float(sqft_living_input.value)
        view = float(view_input.value)
        grade = float(grade_input.value)
        sqft_above = float(sqft_above_input.value)
        sqft_living15 = float(sqft_living15_input.value)

        # Create a numpy array from the inputs
        input_data = np.array([[bathrooms, sqft_living, view, grade, sqft_above, sqft_living15]])
        
        # Standardize the inputs using the saved scaler
        input_data = scaler.transform(input_data)

        # Predict the price using the trained model
        predicted_price = model.predict(input_data)[0]
        result.value = f"Predicted House Price: ${predicted_price:,.2f}"
    except Exception as ex:
        result.value = f"Error: {str(ex)}"
    result.update()

# Define the input fields
bathrooms_input = ft.TextField(label="Number of Bathrooms", width=200)
sqft_living_input = ft.TextField(label="Square Feet Living Area", width=200)
view_input = ft.TextField(label="View", width=200)
grade_input = ft.TextField(label="Grade", width=200)
sqft_above_input = ft.TextField(label="Square Feet Above", width=200)
sqft_living15_input = ft.TextField(label="Square Feet Living 15", width=200)
result = ft.Text()

# Define the predict button
predict_button = ft.ElevatedButton(text="Predict", on_click=predict_price)

# Add the elements to the page
ft.app(
    target=lambda page: page.add(
        bathrooms_input,
        sqft_living_input,
        view_input,
        grade_input,
        sqft_above_input,
        sqft_living15_input,
        predict_button,
        result,
    )
)

# Run the Flet app
ft.app.run()
