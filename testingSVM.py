import numpy as np
import pickle
import pandas as pd

# 1. Define Valid Ranges and Data Types based on the UCI Dataset
# The data types and ranges are crucial for robust input validation.
# We will use dictionaries for categorical mappings.

# Define the boundaries for numerical features
NUMERICAL_RANGES = {
    'Age': (1, 120), # Minimum and Maximum Age in the typical dataset
    'Cholesterol': (126, 564),
    'MaxHR': (60, 202),
    'Oldpeak': (0.0, 6.2)
}

# Define the valid categorical codes and their meanings
CATEGORICAL_MAPPINGS = {
    'Sex': {'Male': 1, 'Female': 0},
    'ChestPainType': {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3},
    'FastingBS': {'< 120 mg/dl (No)': 0, '> 120 mg/dl (Yes)': 1},
    'ExerciseAngina': {'No': 0, 'Yes': 1},
    'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}
}

# The expected feature order for the model
FEATURE_ORDER = ['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']


# 2. Function to Get and Validate Input for a Single Feature
def get_validated_input(feature_name, mapping=None, num_range=None):
    """Prompts user for input and validates it against defined constraints."""
    
    while True:
        try:
            # --- Handle Categorical Input ---
            if mapping:
                options = ', '.join([f"'{k}' ({v})" for k, v in mapping.items()])
                user_input = input(f"Enter {feature_name} ({options}): ").strip()
                
                # Try to map the input (case-insensitive)
                found_key = next((k for k in mapping if k.lower() == user_input.lower()), None)
                
                if found_key is not None:
                    return mapping[found_key] # Return the numerical code
                else:
                    print(f"ERROR: Invalid input. Please enter one of the valid options: {options}")

            # --- Handle Numerical Input ---
            elif num_range:
                min_val, max_val = num_range
                user_input = float(input(f"Enter {feature_name} (Range: {min_val} to {max_val}): "))

                if min_val <= user_input <= max_val:
                    return user_input
                else:
                    print(f"ERROR: Value {user_input} is out of the valid range: {min_val} to {max_val}.")

        except ValueError:
            print("ERROR: Invalid input type. Please enter a number.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


# 3. Main Prediction Function
def predict_heart_disease(model, scaler):
    """Collects input, preprocesses it, and makes a prediction."""
    
    # Storage for the patient's feature values
    patient_data = {}

    print("\n" + "=" * 50)
    print("NEW PATIENT HEART DISEASE PREDICTION")
    print("Please enter the following data for the patient:")
    print("=" * 50)
    
    # Collect all inputs using the validation function
    for feature in FEATURE_ORDER:
        if feature in NUMERICAL_RANGES:
            value = get_validated_input(feature, num_range=NUMERICAL_RANGES[feature])
        elif feature in CATEGORICAL_MAPPINGS:
            value = get_validated_input(feature, mapping=CATEGORICAL_MAPPINGS[feature])
        
        patient_data[feature] = value
    
    print("-" * 50)
    print("Data collected successfully.")

    # Convert to DataFrame in the correct order
    input_df = pd.DataFrame([patient_data], columns=FEATURE_ORDER)
    
    # 4. Preprocessing (Crucial for SVM)
    # The data MUST be scaled using the same scaler fitted during training!
    scaled_data = scaler.transform(input_df)
    
    # 5. Prediction
    prediction = model.predict(scaled_data)
    
    print("=" * 50)
    if prediction[0] == 1:
        print("PREDICTION: HIGH LIKELIHOOD OF HEART DISEASE (Positive)")
    else:
        print("PREDICTION: LOW LIKELIHOOD OF HEART DISEASE (Negative)")
    print("=" * 50)


# 6. Load Model and Scaler, then Run Prediction
if __name__ == "__main__":
    try:
        # Load the trained model and scaler
        with open('heartfail/svm_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
            
        with open('heartfail/scaler.pkl', 'rb') as file:
            loaded_scaler = pickle.load(file)
            
        print("Model and Scaler loaded successfully.")
        
        # Start the interactive prediction process
        predict_heart_disease(loaded_model, loaded_scaler)

    except FileNotFoundError:
        print("\nERROR: Model or Scaler file not found.")
        print("Please ensure you have saved your model as 'svm_model.pkl' and your scaler as 'scaler.pkl'.")
        print("The SVM model relies heavily on scaling, so the scaler is essential.")
    except Exception as e:
        print(f"\nAn error occurred during loading or prediction: {e}")