import numpy as np
import pandas as pd
import pickle
import sys
import os
from datetime import datetime

class HeartDiseasePredictor:
    def __init__(self, model_path=None):
        """
        Initialize the Heart Disease Predictor
        """
        self.features = ['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 
                        'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        
        # Define feature ranges and options based on UCI dataset documentation
        self.feature_info = {
            'Age': {'type': 'int', 'range': (29, 77), 'description': 'Age in years'},
            'Sex': {'type': 'categorical', 'options': ['Male', 'Female'], 
                   'mapping': {'Male': 1, 'Female': 0}},
            'ChestPainType': {'type': 'categorical', 'options': 
                             ['Typical Angina', 'Atypical Angina', 
                              'Non-anginal Pain', 'Asymptomatic'],
                             'mapping': {'Typical Angina': 1, 'Atypical Angina': 2,
                                        'Non-anginal Pain': 3, 'Asymptomatic': 4}},
            'Cholesterol': {'type': 'int', 'range': (126, 564), 
                           'description': 'Serum cholesterol in mg/dl'},
            'FastingBS': {'type': 'binary', 'options': ['≤120 mg/dl', '>120 mg/dl'],
                         'mapping': {'≤120 mg/dl': 0, '>120 mg/dl': 1}},
            'MaxHR': {'type': 'int', 'range': (71, 202), 
                     'description': 'Maximum heart rate achieved'},
            'ExerciseAngina': {'type': 'binary', 'options': ['No', 'Yes'],
                              'mapping': {'No': 0, 'Yes': 1}},
            'Oldpeak': {'type': 'float', 'range': (0.0, 6.2), 
                       'description': 'ST depression induced by exercise relative to rest'},
            'ST_Slope': {'type': 'categorical', 'options': 
                        ['Upsloping', 'Flat', 'Downsloping'],
                        'mapping': {'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}}
        }
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained SVM model and preprocessing objects"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.label_encoder = model_data.get('label_encoder')
                print(f"✓ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("⚠️ Continuing with manual input mode...")
    
    def display_welcome(self):
        """Display welcome message and instructions"""
        print("="*60)
        print("HEART DISEASE PREDICTION SYSTEM")
        print("="*60)
        print("\nBased on UCI Heart Disease Dataset (Cleveland Database)")
        print("Features extracted from: Age, Sex, ChestPainType, Cholesterol,")
        print("FastingBS, MaxHR, ExerciseAngina, Oldpeak, ST_Slope")
        print("\n" + "-"*60)
    
    def get_numeric_input(self, feature_name, input_type='int'):
        """
        Get numeric input with validation
        
        Parameters:
        -----------
        feature_name : str
            Name of the feature
        input_type : str
            Type of input ('int' or 'float')
        """
        info = self.feature_info[feature_name]
        min_val, max_val = info['range']
        
        while True:
            try:
                print(f"\n→ {feature_name}: {info['description']}")
                print(f"   Range: {min_val} to {max_val}")
                
                prompt = f"   Enter {feature_name.lower()} ({min_val}-{max_val}): "
                user_input = input(prompt).strip()
                
                if not user_input:
                    print("   ⚠️  Input cannot be empty. Please try again.")
                    continue
                
                if input_type == 'int':
                    value = int(user_input)
                else:  # float
                    value = float(user_input)
                
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"   ⚠️  Value must be between {min_val} and {max_val}. Try again.")
                    
            except ValueError:
                print(f"   ⚠️  Invalid input. Please enter a valid {input_type}.")
            except KeyboardInterrupt:
                print("\n\n⚠️  Input cancelled by user.")
                sys.exit(0)
    
    def get_categorical_input(self, feature_name, display_as_list=True):
        """
        Get categorical input with serial number selection
        
        Parameters:
        -----------
        feature_name : str
            Name of the feature
        display_as_list : bool
            Whether to display options as numbered list
        """
        info = self.feature_info[feature_name]
        options = info['options']
        mapping = info['mapping']
        
        while True:
            try:
                print(f"\n→ {feature_name}:")
                
                if display_as_list:
                    print("   Please select from the following options:")
                    for i, option in enumerate(options, 1):
                        print(f"   {i}. {option}")
                    
                    choice = input(f"   Enter your choice (1-{len(options)}): ").strip()
                    
                    if not choice.isdigit():
                        print("   ⚠️  Please enter a valid number.")
                        continue
                    
                    choice_idx = int(choice) - 1
                    
                    if 0 <= choice_idx < len(options):
                        selected_option = options[choice_idx]
                        return mapping[selected_option]
                    else:
                        print(f"   ⚠️  Choice must be between 1 and {len(options)}.")
                else:
                    # For binary choices, use simpler input
                    print(f"   Options: {', '.join(options)}")
                    user_input = input(f"   Enter your choice: ").strip()
                    
                    # Try to match input with options
                    for option in options:
                        if user_input.lower() in option.lower() or \
                           (len(user_input) > 0 and option.lower().startswith(user_input.lower())):
                            return mapping[option]
                    
                    print(f"   ⚠️  Invalid choice. Please enter one of: {', '.join(options)}")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Input cancelled by user.")
                sys.exit(0)
    
    def get_binary_input(self, feature_name):
        """Get binary input (Yes/No type)"""
        info = self.feature_info[feature_name]
        options = info['options']
        
        while True:
            try:
                print(f"\n→ {feature_name}:")
                print(f"   Options: {options[0]} (Enter: 0) | {options[1]} (Enter: 1)")
                
                choice = input(f"   Enter 0 or 1: ").strip()
                
                if choice in ['0', '1']:
                    return int(choice)
                else:
                    print("   ⚠️  Please enter either 0 or 1.")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Input cancelled by user.")
                sys.exit(0)
    
    def collect_user_inputs(self):
        """Collect all feature inputs from user"""
        print("\n" + "="*60)
        print("PATIENT DATA ENTRY")
        print("="*60)
        
        user_data = {}
        
        # Collect each feature input
        for feature in self.features:
            info = self.feature_info[feature]
            
            if info['type'] == 'int':
                user_data[feature] = self.get_numeric_input(feature, 'int')
            elif info['type'] == 'float':
                user_data[feature] = self.get_numeric_input(feature, 'float')
            elif info['type'] == 'categorical':
                user_data[feature] = self.get_categorical_input(feature)
            elif info['type'] == 'binary':
                user_data[feature] = self.get_binary_input(feature)
        
        return user_data
    
    def validate_inputs(self, user_data):
        """Validate all collected inputs"""
        print("\n" + "="*60)
        print("INPUT VALIDATION")
        print("="*60)
        
        print("\nYou have entered the following values:")
        for feature, value in user_data.items():
            info = self.feature_info[feature]
            
            if info['type'] in ['int', 'float']:
                print(f"  {feature}: {value} ({info['description']})")
            elif info['type'] == 'categorical':
                # Find the key from mapping
                mapping = info['mapping']
                for option, map_val in mapping.items():
                    if map_val == value:
                        print(f"  {feature}: {option}")
                        break
            elif info['type'] == 'binary':
                options = info['options']
                mapping = info['mapping']
                for option, map_val in mapping.items():
                    if map_val == value:
                        print(f"  {feature}: {option}")
                        break
        
        while True:
            confirm = input("\nAre these values correct? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                return True
            elif confirm in ['no', 'n']:
                print("\nLet's re-enter the data...")
                return False
            else:
                print("Please enter 'yes' or 'no'.")
    
    def preprocess_inputs(self, user_data):
        """Preprocess user inputs for model prediction"""
        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame([user_data])[self.features]
        
        # Apply preprocessing if scaler is available
        if self.scaler is not None:
            input_scaled = self.scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        return input_scaled
    
    def predict(self, input_data):
        """Make prediction using SVM model"""
        if self.model is None:
            print("\n⚠️  No model loaded. Using rule-based estimation instead.")
            return self.rule_based_estimation(input_data[0])
        
        try:
            prediction = self.model.predict(input_data)
            probability = self.model.predict_proba(input_data)
            
            return {
                'prediction': prediction[0],
                'probability': probability[0],
                'confidence': max(probability[0])
            }
        except Exception as e:
            print(f"✗ Prediction error: {e}")
            return self.rule_based_estimation(input_data[0])
    
    def rule_based_estimation(self, features):
        """Rule-based estimation when model is not available"""
        # Simple risk estimation based on clinical knowledge
        risk_score = 0
        
        # Age risk (increased risk over 45)
        if features[0] > 45:  # Age index
            risk_score += 1
        
        # Cholesterol risk (>200 mg/dl)
        if features[3] > 200:  # Cholesterol index
            risk_score += 1
        
        # Max HR risk (lower HR indicates higher risk)
        if features[5] < 120:  # MaxHR index
            risk_score += 1
        
        # Oldpeak risk (ST depression)
        if features[7] > 1.0:  # Oldpeak index
            risk_score += 1
        
        # Estimate probability based on risk score
        probability = min(0.95, risk_score * 0.25)
        prediction = 1 if probability > 0.5 else 0
        
        return {
            'prediction': prediction,
            'probability': [1-probability, probability],
            'confidence': probability if prediction == 1 else 1-probability
        }
    
    def display_results(self, result):
        """Display prediction results"""
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        
        if result['prediction'] == 0:
            print("\n✅ RESULT: LOW RISK OF HEART DISEASE")
            print(f"   Confidence: {result['confidence']*100:.1f}%")
            print("\n   The model predicts NO significant heart disease.")
            print("   (Angiographic diameter narrowing < 50%)")
        else:
            print("\n⚠️ RESULT: HIGH RISK OF HEART DISEASE")
            print(f"   Confidence: {result['confidence']*100:.1f}%")
            print("\n   The model predicts PRESENCE of heart disease.")
            print("   (Angiographic diameter narrowing > 50%)")
        
        print("\n" + "-"*60)
        print("PROBABILITY DISTRIBUTION:")
        print(f"   No Heart Disease: {result['probability'][0]*100:.1f}%")
        print(f"   Heart Disease:    {result['probability'][1]*100:.1f}%")
        
        print("\n" + "-"*60)
        print("IMPORTANT NOTES:")
        print("1. This prediction is based on machine learning algorithms")
        print("2. Always consult with healthcare professionals")
        print("3. This tool is for educational/research purposes only")
        print("="*60)
    
    def save_inputs_to_file(self, user_data, result, filename="heart_disease_predictions.csv"):
        """Save inputs and results to CSV file"""
        try:
            # Create results dictionary
            results_dict = user_data.copy()
            results_dict['Prediction'] = 'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'
            results_dict['Confidence'] = f"{result['confidence']*100:.1f}%"
            results_dict['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Convert to DataFrame
            results_df = pd.DataFrame([results_dict])
            
            # Check if file exists
            if os.path.exists(filename):
                # Append to existing file
                results_df.to_csv(filename, mode='a', header=False, index=False)
            else:
                # Create new file
                results_df.to_csv(filename, index=False)
            
            print(f"\n📊 Results saved to {filename}")
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")
    
    def run(self):
        """Main execution function"""
        self.display_welcome()
        
        while True:
            # Collect user inputs
            user_data = self.collect_user_inputs()
            
            # Validate inputs
            if not self.validate_inputs(user_data):
                continue
            
            # Preprocess inputs
            processed_data = self.preprocess_inputs(user_data)
            
            # Make prediction
            result = self.predict(processed_data)
            
            # Display results
            self.display_results(result)
            
            # Save to file
            self.save_inputs_to_file(user_data, result)
            
            # Ask for another prediction
            print("\n" + "="*60)
            another = input("Would you like to make another prediction? (yes/no): ").strip().lower()
            if another not in ['yes', 'y']:
                print("\nThank you for using the Heart Disease Prediction System!")
                print("="*60)
                break


def main():
    """Main function to run the predictor"""
    
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Check if model file exists
    model_files = ['heartfail/heart_disease_svm_model.pkl']
    found_model = None
    
    for model_file in model_files:
        if os.path.exists(model_file):
            found_model = model_file
            break
    
    if found_model:
        use_model = input(f"Found model file: {found_model}. Use it? (yes/no): ").strip().lower()
        if use_model in ['yes', 'y']:
            predictor.load_model(found_model)
    
    # Run the prediction system
    try:
        predictor.run()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your inputs and try again.")


if __name__ == "__main__":
    main()