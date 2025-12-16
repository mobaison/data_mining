from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import csv
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = '13680'

# Static storage configuration
DATA_DIR = Path("heartfail/ui/data")
PREDICTIONS_FILE = DATA_DIR / "predictions.csv"
DATA_DIR.mkdir(exist_ok=True)

# Initialize predictions file if it doesn't exist
if not PREDICTIONS_FILE.exists():
    with open(PREDICTIONS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'id', 'patient_name', 'age', 'sex', 'chest_pain_type',
            'cholesterol', 'fasting_bs', 'max_hr', 'exercise_angina',
            'oldpeak', 'st_slope', 'prediction_result', 'confidence',
            'risk_level', 'timestamp'
        ])

class StaticStorage:
    """Static storage manager for predictions"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self._predictions = None
    
    def load_predictions(self):
        """Load predictions from CSV file"""
        if self._predictions is None:
            try:
                # Check if file is empty (only headers)
                with open(self.file_path, 'r') as f:
                    content = f.read().strip()
                    if not content or len(content.split('\n')) <= 1:
                        self._predictions = pd.DataFrame()
                        return self._predictions
                
                # Load the CSV
                df = pd.read_csv(self.file_path)
                
                # Convert numeric columns
                numeric_columns = ['id', 'age', 'cholesterol', 'max_hr', 'confidence']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Convert oldpeak to float
                if 'oldpeak' in df.columns:
                    df['oldpeak'] = pd.to_numeric(df['oldpeak'], errors='coerce')
                
                # Convert timestamp
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                self._predictions = df
            except Exception as e:
                print(f"Error loading predictions: {e}")
                self._predictions = pd.DataFrame()
        
        return self._predictions
    
    def save_prediction(self, prediction_data):
        """Save a new prediction to CSV file"""
        try:
            # Generate ID
            existing_df = self.load_predictions()
            new_id = 1
            if not existing_df.empty and 'id' in existing_df.columns:
                max_id = existing_df['id'].max()
                if pd.notna(max_id):
                    new_id = int(max_id) + 1
            
            # Add ID and timestamp
            prediction_data['id'] = new_id
            prediction_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Convert to DataFrame row
            new_row = pd.DataFrame([prediction_data])
            
            # Append to file
            with open(self.file_path, 'a', newline='') as f:
                new_row.to_csv(f, header=False, index=False)
            
            # Clear cache to reload on next access
            self._predictions = None
            
            return new_id
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return None
    
    def get_predictions(self, limit=None):
        """Get all predictions with optional limit"""
        df = self.load_predictions()
        if df.empty:
            return []
        
        # Sort by timestamp descending
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp', ascending=False)
        
        if limit:
            return df.head(limit).to_dict('records')
        return df.to_dict('records')
    
    def delete_prediction(self, prediction_id):
        """Delete a prediction by ID"""
        try:
            df = self.load_predictions()
            if df.empty:
                return False
            
            df = df[df['id'] != prediction_id]
            df.to_csv(self.file_path, index=False)
            self._predictions = None
            return True
        except Exception as e:
            print(f"Error deleting prediction: {e}")
            return False
    
    def get_stats(self):
        """Calculate statistics from predictions"""
        df = self.load_predictions()
        
        if df.empty:
            return {
                'total_predictions': 0,
                'high_risk': 0,
                'average_age': 0,
                'average_cholesterol': 0,
                'male_count': 0,
                'female_count': 0,
                'avg_confidence': 0
            }
        
        stats = {
            'total_predictions': len(df),
            'high_risk': 0,
            'average_age': 0,
            'average_cholesterol': 0,
            'male_count': 0,
            'female_count': 0,
            'avg_confidence': 0
        }
        
        # Calculate high risk count safely
        if 'risk_level' in df.columns:
            stats['high_risk'] = len(df[df['risk_level'] == 'High'])
        
        # Calculate average age safely
        if 'age' in df.columns and df['age'].notna().any():
            stats['average_age'] = round(df['age'].mean(skipna=True), 1)
        
        # Calculate average cholesterol safely
        if 'cholesterol' in df.columns and df['cholesterol'].notna().any():
            stats['average_cholesterol'] = round(df['cholesterol'].mean(skipna=True), 0)
        
        # Calculate gender counts safely
        if 'sex' in df.columns:
            stats['male_count'] = len(df[df['sex'] == 'Male'])
            stats['female_count'] = len(df[df['sex'] == 'Female'])
        
        # Calculate average confidence safely
        if 'confidence' in df.columns and df['confidence'].notna().any():
            stats['avg_confidence'] = round(df['confidence'].mean(skipna=True), 1)
        
        return stats

# Initialize storage
storage = StaticStorage(PREDICTIONS_FILE)

# Heart Disease Predictor Class
class HeartDiseasePredictor:
    def __init__(self):
        self.features = ['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 
                        'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        
        # Define feature info - NO AGE RANGE RESTRICTION
        self.feature_info = {
            'Age': {'type': 'int', 'description': 'Age in years'},
            'Sex': {'type': 'categorical', 'options': ['Male', 'Female'], 
                   'mapping': {'Male': 1, 'Female': 0}},
            'ChestPainType': {'type': 'categorical', 'options': 
                             ['Typical Angina', 'Atypical Angina', 
                              'Non-anginal Pain', 'Asymptomatic'],
                             'mapping': {'Typical Angina': 1, 'Atypical Angina': 2,
                                        'Non-anginal Pain': 3, 'Asymptomatic': 4}},
            'Cholesterol': {'type': 'int', 'range': (100, 600), 
                           'description': 'Serum cholesterol in mg/dl'},
            'FastingBS': {'type': 'binary', 'options': ['≤120 mg/dl', '>120 mg/dl'],
                         'mapping': {'≤120 mg/dl': 0, '>120 mg/dl': 1}},
            'MaxHR': {'type': 'int', 'range': (60, 220), 
                     'description': 'Maximum heart rate achieved'},
            'ExerciseAngina': {'type': 'binary', 'options': ['No', 'Yes'],
                              'mapping': {'No': 0, 'Yes': 1}},
            'Oldpeak': {'type': 'float', 'range': (0.0, 10.0), 
                       'description': 'ST depression induced by exercise relative to rest'},
            'ST_Slope': {'type': 'categorical', 'options': 
                        ['Upsloping', 'Flat', 'Downsloping'],
                        'mapping': {'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}}
        }
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        # Try to load model from multiple possible locations
        self.load_model()
    
    def load_model(self):
        """Load trained SVM model from multiple possible locations"""
        model_paths = [
            'heartfail/ui/models/heart_disease_svm_model.pkll',
            'heart_disease_svm_model.pkl',
            'heartfail/heart_disease_svm_model.pkl',
            '../heart_disease_svm_model.pkl'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        self.model = model_data.get('model')
                        self.scaler = model_data.get('scaler')
                        self.label_encoder = model_data.get('label_encoder')
                        print(f"✓ Model loaded successfully from {model_path}")
                        return True
                except Exception as e:
                    print(f"✗ Error loading model from {model_path}: {e}")
        
        print("⚠️ No model found. Using rule-based estimation instead.")
        return False
    
    def preprocess_inputs(self, user_data):
        """Preprocess user inputs for model prediction"""
        # Convert user data to correct format
        feature_values = []
        
        # Age
        feature_values.append(int(user_data['age']))
        
        # Sex
        sex_mapping = {'Male': 1, 'Female': 0}
        feature_values.append(sex_mapping.get(user_data['sex'], 1))
        
        # ChestPainType
        cp_mapping = {'Typical Angina': 1, 'Atypical Angina': 2, 
                     'Non-anginal Pain': 3, 'Asymptomatic': 4}
        feature_values.append(cp_mapping.get(user_data['chest_pain_type'], 1))
        
        # Cholesterol
        feature_values.append(int(user_data['cholesterol']))
        
        # FastingBS
        fbs_mapping = {'≤120 mg/dl': 0, '>120 mg/dl': 1}
        feature_values.append(fbs_mapping.get(user_data['fasting_bs'], 0))
        
        # MaxHR
        feature_values.append(int(user_data['max_hr']))
        
        # ExerciseAngina
        exang_mapping = {'No': 0, 'Yes': 1}
        feature_values.append(exang_mapping.get(user_data['exercise_angina'], 0))
        
        # Oldpeak
        feature_values.append(float(user_data['oldpeak']))
        
        # ST_Slope
        slope_mapping = {'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}
        feature_values.append(slope_mapping.get(user_data['st_slope'], 1))
        
        # Convert to numpy array and reshape
        input_array = np.array(feature_values).reshape(1, -1)
        
        # Apply preprocessing if scaler is available
        if self.scaler is not None:
            input_scaled = self.scaler.transform(input_array)
        else:
            input_scaled = input_array
        
        return input_scaled
    
    def predict(self, input_data):
        """Make prediction using SVM model"""
        try:
            if self.model is None:
                print("⚠️ No model loaded. Using rule-based estimation instead.")
                return self.rule_based_estimation(input_data[0])
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            
            # Get probabilities if available
            try:
                probability = self.model.predict_proba(input_data)[0]
                confidence = max(probability)
            except:
                # If predict_proba is not available, use prediction as probability
                probability = [1-prediction, prediction] if prediction in [0, 1] else [0.5, 0.5]
                confidence = 0.8 if prediction in [0, 1] else 0.5
            
            # Convert numpy types to Python native types for JSON serialization
            probability = probability.tolist() if hasattr(probability, 'tolist') else probability
            prediction = int(prediction)
            confidence = float(confidence)
            
            # Map prediction to result
            if prediction == 0:
                result = "No Heart Disease"
                risk_level = "Low"
            else:
                result = "Heart Disease Detected"
                risk_level = "High" if confidence > 0.7 else "Moderate"
            
            return {
                'result': result,
                'confidence': round(confidence * 100, 2),
                'risk_level': risk_level,
                'probability': probability,
                'prediction': prediction,
                'details': []
            }
            
        except Exception as e:
            print(f"✗ Prediction error in SVM model: {e}")
            # Fall back to rule-based estimation
            return self.rule_based_estimation(input_data[0])
    
    def rule_based_estimation(self, features):
        """Rule-based estimation when model is not available"""
        try:
            # Extract features (ensure they're in the right order)
            # Order: Age, Sex, ChestPainType, Cholesterol, FastingBS, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
            
            risk_score = 0
            risk_factors = []
            
            # Age risk (increased risk over 45) - Use index 0
            age = features[0]
            if age > 45:
                risk_score += 1
                risk_factors.append({
                    'factor': 'Age',
                    'level': 'Moderate',
                    'description': f'Age ({age}) is above 45 years'
                })
            
            # Cholesterol risk (>200 mg/dl) - Use index 3
            cholesterol = features[3]
            if cholesterol > 200:
                risk_score += 1
                risk_factors.append({
                    'factor': 'Cholesterol',
                    'level': 'High' if cholesterol > 240 else 'Moderate',
                    'description': f'Cholesterol level ({cholesterol}) is above 200 mg/dl'
                })
            
            # Max HR risk (lower HR indicates higher risk) - Use index 5
            max_hr = features[5]
            if max_hr < 120:
                risk_score += 1
                risk_factors.append({
                    'factor': 'Maximum Heart Rate',
                    'level': 'Moderate',
                    'description': f'Maximum heart rate ({max_hr}) is below 120 bpm'
                })
            
            # Oldpeak risk (ST depression) - Use index 7
            oldpeak = features[7]
            if oldpeak > 1.0:
                risk_score += 1
                level = 'High' if oldpeak > 2.0 else 'Moderate'
                risk_factors.append({
                    'factor': 'ST Depression',
                    'level': level,
                    'description': f'ST depression ({oldpeak}) is above 1.0'
                })
            
            # Chest Pain Type risk - Use index 2
            # Values: 1=Typical Angina, 2=Atypical Angina, 3=Non-anginal, 4=Asymptomatic
            cp_type = features[2]
            if cp_type == 4:  # Asymptomatic
                risk_score += 1
                risk_factors.append({
                    'factor': 'Chest Pain Type',
                    'level': 'High',
                    'description': 'Asymptomatic chest pain (can indicate silent ischemia)'
                })
            
            # Exercise Angina risk - Use index 6
            # Values: 0=No, 1=Yes
            exang = features[6]
            if exang == 1:
                risk_score += 1
                risk_factors.append({
                    'factor': 'Exercise Angina',
                    'level': 'High',
                    'description': 'Exercise-induced angina present'
                })
            
            # Estimate probability based on risk score
            probability = min(0.95, risk_score * 0.25)
            
            if probability > 0.5:
                result = "Heart Disease Detected"
                risk_level = "High" if probability > 0.7 else "Moderate"
                confidence = probability
                prediction = 1
            else:
                result = "No Heart Disease"
                risk_level = "Low"
                confidence = 1 - probability
                prediction = 0
            
            # Convert to Python native types
            probability = [float(1-probability), float(probability)]
            confidence = float(confidence)
            
            return {
                'result': result,
                'confidence': round(confidence * 100, 2),
                'risk_level': risk_level,
                'probability': probability,
                'prediction': prediction,
                'details': risk_factors
            }
            
        except Exception as e:
            print(f"✗ Rule-based estimation error: {e}")
            # Return safe default with native Python types
            return {
                'result': "No Heart Disease",
                'confidence': 50.0,
                'risk_level': "Low",
                'probability': [0.5, 0.5],
                'prediction': 0,
                'details': []
            }
    
    def get_risk_factors(self, input_data):
        """Identify key risk factors from input data"""
        factors = []
        
        # Extract values safely
        try:
            age = int(input_data.get('age', 0))
            cholesterol = int(input_data.get('cholesterol', 0))
            max_hr = int(input_data.get('max_hr', 0))
            oldpeak = float(input_data.get('oldpeak', 0))
            chest_pain_type = input_data.get('chest_pain_type', '')
            exercise_angina = input_data.get('exercise_angina', 'No')
            
            # High cholesterol
            if cholesterol > 240:
                factors.append({
                    'factor': 'High Cholesterol',
                    'level': 'High',
                    'description': f'Cholesterol level ({cholesterol}) above 240 mg/dl significantly increases heart disease risk'
                })
            elif cholesterol > 200:
                factors.append({
                    'factor': 'Elevated Cholesterol',
                    'level': 'Moderate',
                    'description': f'Cholesterol level ({cholesterol}) above 200 mg/dl increases heart disease risk'
                })
            
            # Low max heart rate
            if max_hr < 100:
                factors.append({
                    'factor': 'Low Maximum Heart Rate',
                    'level': 'High',
                    'description': f'Maximum heart rate ({max_hr}) below 100 bpm may indicate poor cardiovascular fitness'
                })
            elif max_hr < 120:
                factors.append({
                    'factor': 'Below Average Heart Rate',
                    'level': 'Moderate',
                    'description': f'Maximum heart rate ({max_hr}) below 120 bpm'
                })
            
            # High ST depression
            if oldpeak > 2.0:
                factors.append({
                    'factor': 'High ST Depression',
                    'level': 'High',
                    'description': f'ST depression ({oldpeak}) > 2.0 indicates potential myocardial ischemia'
                })
            elif oldpeak > 1.0:
                factors.append({
                    'factor': 'Elevated ST Depression',
                    'level': 'Moderate',
                    'description': f'ST depression ({oldpeak}) > 1.0'
                })
            
            # Asymptomatic chest pain
            if chest_pain_type == 'Asymptomatic':
                factors.append({
                    'factor': 'Asymptomatic Chest Pain',
                    'level': 'High',
                    'description': 'Asymptomatic presentation can indicate silent ischemia'
                })
            
            # Exercise angina
            if exercise_angina == 'Yes':
                factors.append({
                    'factor': 'Exercise-Induced Angina',
                    'level': 'High',
                    'description': 'Chest pain during exercise indicates potential coronary artery disease'
                })
            
            # Age factor
            if age > 65:
                factors.append({
                    'factor': 'Advanced Age',
                    'level': 'High',
                    'description': f'Age ({age}) is a significant risk factor for heart disease'
                })
            elif age > 45:
                factors.append({
                    'factor': 'Age',
                    'level': 'Moderate',
                    'description': f'Age ({age}) increases heart disease risk'
                })
                
        except Exception as e:
            print(f"Error extracting risk factors: {e}")
        
        return factors

# Initialize predictor
predictor = HeartDiseasePredictor()

# Helper function to convert numpy types to Python native types
def convert_to_serializable(obj):
    """Convert numpy and other non-serializable types to Python native types"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# Routes
@app.route('/')
def index():
    """Home page"""
    try:
        stats = storage.get_stats()
        recent_predictions = storage.get_predictions(limit=5)
        
        return render_template('index.html', 
                             total_predictions=stats['total_predictions'],
                             high_risk=stats['high_risk'],
                             recent_predictions=recent_predictions)
    except Exception as e:
        print(f"Error in index route: {e}")
        return render_template('index.html', 
                             total_predictions=0,
                             high_risk=0,
                             recent_predictions=[])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'POST':
        try:
            # Get form data
            patient_data = {
                'patient_name': request.form.get('patient_name', '').strip(),
                'age': request.form.get('age', '').strip(),
                'sex': request.form.get('sex', '').strip(),
                'chest_pain_type': request.form.get('chest_pain_type', '').strip(),
                'cholesterol': request.form.get('cholesterol', '').strip(),
                'fasting_bs': request.form.get('fasting_bs', '').strip(),
                'max_hr': request.form.get('max_hr', '').strip(),
                'exercise_angina': request.form.get('exercise_angina', '').strip(),
                'oldpeak': request.form.get('oldpeak', '').strip(),
                'st_slope': request.form.get('st_slope', '').strip()
            }
            
            # Debug: print received data
            print("Received patient data:", patient_data)
            
            # Validate required fields
            required_fields = ['patient_name', 'age', 'sex', 'chest_pain_type', 
                              'cholesterol', 'fasting_bs', 'max_hr', 
                              'exercise_angina', 'oldpeak', 'st_slope']
            
            missing_fields = []
            for field in required_fields:
                if not patient_data[field]:
                    missing_fields.append(field.replace('_', ' ').title())
            
            if missing_fields:
                error_msg = f"The following fields are required: {', '.join(missing_fields)}"
                print(f"Validation error: {error_msg}")
                return jsonify({'error': error_msg})
            
            # Convert numeric fields
            try:
                patient_data['age'] = int(patient_data['age'])
                patient_data['cholesterol'] = int(patient_data['cholesterol'])
                patient_data['max_hr'] = int(patient_data['max_hr'])
                patient_data['oldpeak'] = float(patient_data['oldpeak'])
            except ValueError as e:
                error_msg = f"Invalid numeric value: {str(e)}"
                print(f"Validation error: {error_msg}")
                return jsonify({'error': error_msg})
            
            # Make prediction
            print("Making prediction...")
            
            # Preprocess input
            processed_data = predictor.preprocess_inputs(patient_data)
            print(f"Processed data shape: {processed_data.shape}")
            print(f"Processed data: {processed_data}")
            
            # Get prediction
            result = predictor.predict(processed_data)
            print(f"Prediction result: {result}")
            
            # Get additional risk factors
            risk_factors = predictor.get_risk_factors(patient_data)
            if risk_factors:
                result['details'] = risk_factors
            
            # Convert result to serializable format
            result = convert_to_serializable(result)
            
            # Prepare data for storage
            prediction_to_save = {
                'patient_name': patient_data['patient_name'],
                'age': patient_data['age'],
                'sex': patient_data['sex'],
                'chest_pain_type': patient_data['chest_pain_type'],
                'cholesterol': patient_data['cholesterol'],
                'fasting_bs': patient_data['fasting_bs'],
                'max_hr': patient_data['max_hr'],
                'exercise_angina': patient_data['exercise_angina'],
                'oldpeak': patient_data['oldpeak'],
                'st_slope': patient_data['st_slope'],
                'prediction_result': result['result'],
                'confidence': result['confidence'],
                'risk_level': result['risk_level']
            }
            
            # Save to static storage
            prediction_id = storage.save_prediction(prediction_to_save)
            print(f"Saved prediction with ID: {prediction_id}")
            
            # Store in session for results page (make sure everything is serializable)
            session['prediction_result'] = result
            session['patient_data'] = patient_data
            session['prediction_id'] = prediction_id
            
            # Force session save
            session.modified = True
            
            return redirect(url_for('results'))
            
        except Exception as e:
            error_msg = f"Server error: {str(e)}"
            print(f"Error in predict route: {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': error_msg})
    
    return render_template('predict.html', feature_info=predictor.feature_info)

@app.route('/results')
def results():
    """Results page"""
    result = session.get('prediction_result')
    patient_data = session.get('patient_data')
    prediction_id = session.get('prediction_id')
    
    if not result or not patient_data:
        print("No prediction data in session, redirecting to predict page")
        return redirect(url_for('predict'))
    
    print(f"Displaying results for prediction ID: {prediction_id}")
    print(f"Result data: {result}")
    
    return render_template('results.html', 
                         result=result, 
                         patient_data=patient_data,
                         prediction_id=prediction_id)

@app.route('/dashboard')
def dashboard():
    """Dashboard with analytics"""
    try:
        # Get all predictions
        predictions = storage.get_predictions()
        
        if not predictions:
            return render_template('dashboard.html', 
                                 age_chart=None, 
                                 risk_chart=None,
                                 trend_chart=None,
                                 stats={})
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(predictions)
        
        # Create visualizations if we have data
        charts = {}
        
        # Age distribution
        if 'age' in df.columns and len(df) > 0:
            age_fig = px.histogram(df, x='age', nbins=20, 
                                  title='Age Distribution of Patients',
                                  labels={'age': 'Age', 'count': 'Number of Patients'},
                                  color_discrete_sequence=['#6366f1'])
            charts['age_chart'] = json.dumps(age_fig, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            charts['age_chart'] = None
        
        # Risk level distribution
        if 'risk_level' in df.columns and len(df) > 0:
            risk_counts = df['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['risk_level', 'count']
            risk_fig = px.pie(risk_counts, values='count', names='risk_level',
                             title='Risk Level Distribution',
                             color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444'])
            charts['risk_chart'] = json.dumps(risk_fig, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            charts['risk_chart'] = None
        
        # Daily trend
        if 'timestamp' in df.columns and len(df) > 0:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                daily_trend = df.groupby(df['timestamp'].dt.date).size().reset_index()
                daily_trend.columns = ['date', 'count']
                trend_fig = px.line(daily_trend, x='date', y='count',
                                   title='Daily Prediction Trends',
                                   labels={'date': 'Date', 'count': 'Number of Predictions'},
                                   color_discrete_sequence=['#8b5cf6'])
                charts['trend_chart'] = json.dumps(trend_fig, cls=plotly.utils.PlotlyJSONEncoder)
            except:
                charts['trend_chart'] = None
        else:
            charts['trend_chart'] = None
        
        # Get statistics
        stats = storage.get_stats()
        
        return render_template('dashboard.html',
                             age_chart=charts['age_chart'],
                             risk_chart=charts['risk_chart'],
                             trend_chart=charts['trend_chart'],
                             stats=stats)
    except Exception as e:
        print(f"Error in dashboard route: {e}")
        return render_template('dashboard.html',
                             age_chart=None,
                             risk_chart=None,
                             trend_chart=None,
                             stats={})

@app.route('/api/predictions')
def get_predictions():
    """API endpoint to get prediction history"""
    try:
        predictions = storage.get_predictions(limit=50)
        
        # Format timestamps if they exist
        formatted_predictions = []
        for pred in predictions:
            formatted_pred = pred.copy()
            if 'timestamp' in formatted_pred and formatted_pred['timestamp']:
                try:
                    # Try to format timestamp
                    if isinstance(formatted_pred['timestamp'], str):
                        dt = datetime.strptime(formatted_pred['timestamp'], '%Y-%m-%d %H:%M:%S')
                    else:
                        dt = formatted_pred['timestamp']
                    formatted_pred['timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_pred['timestamp'] = str(formatted_pred['timestamp'])
            formatted_predictions.append(formatted_pred)
        
        return jsonify(formatted_predictions)
    except Exception as e:
        print(f"Error in get_predictions API: {e}")
        return jsonify([])

@app.route('/api/delete/<int:prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    """Delete a prediction record"""
    try:
        success = storage.delete_prediction(prediction_id)
        if success:
            return jsonify({'success': True})
        return jsonify({'error': 'Prediction not found'}), 404
    except Exception as e:
        print(f"Error in delete_prediction API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<format_type>')
def export_data(format_type):
    """Export predictions in specified format"""
    try:
        predictions = storage.get_predictions()
        
        if format_type == 'csv':
            # Convert to CSV
            df = pd.DataFrame(predictions)
            csv_data = df.to_csv(index=False)
            
            response = app.response_class(
                response=csv_data,
                mimetype='text/csv',
                headers={'Content-disposition': 'attachment; filename=predictions.csv'}
            )
            return response
        
        elif format_type == 'json':
            # Convert to JSON
            return jsonify(predictions)
        
        return jsonify({'error': 'Invalid format'}), 400
    except Exception as e:
        print(f"Error in export_data API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats_api():
    """API endpoint to get statistics"""
    try:
        stats = storage.get_stats()
        return jsonify(stats)
    except Exception as e:
        print(f"Error in get_stats API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_status')
def get_model_status():
    """Check if model is loaded"""
    try:
        model_loaded = predictor.model is not None
        return jsonify({
            'model_loaded': model_loaded,
            'model_type': type(predictor.model).__name__ if model_loaded else None,
            'scaler_loaded': predictor.scaler is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Custom JSON encoder to handle numpy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Apply the custom encoder
app.json_encoder = NumpyJSONEncoder

if __name__ == '__main__':
    # Debug information on startup
    print("="*60)
    print("HEART DISEASE PREDICTION SYSTEM - Flask Web Application")
    print("="*60)
    print(f"Storage file: {PREDICTIONS_FILE}")
    print(f"File exists: {PREDICTIONS_FILE.exists()}")
    
    # Check model loading
    print("\nModel loading status:")
    if predictor.model:
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(predictor.model).__name__}")
        print(f"  Scaler available: {'Yes' if predictor.scaler else 'No'}")
        print(f"  Features used: {predictor.features}")
    else:
        print("⚠️ No model loaded - using rule-based prediction")
    
    print("\nStarting Flask server on http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, port=5002)