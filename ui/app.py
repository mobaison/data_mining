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
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Static storage configuration
DATA_DIR = Path("ui/data")
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
                df = pd.read_csv(self.file_path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
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
            new_id = existing_df['id'].max() + 1 if not existing_df.empty else 1
            
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
        if limit:
            return df.tail(limit).to_dict('records')
        return df.to_dict('records')
    
    def delete_prediction(self, prediction_id):
        """Delete a prediction by ID"""
        try:
            df = self.load_predictions()
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
        
        return {
            'total_predictions': len(df),
            'high_risk': len(df[df['risk_level'] == 'High']) if 'risk_level' in df.columns else 0,
            'average_age': round(df['age'].mean(), 1) if 'age' in df.columns else 0,
            'average_cholesterol': round(df['cholesterol'].mean(), 0) if 'cholesterol' in df.columns else 0,
            'male_count': len(df[df['sex'] == 'Male']) if 'sex' in df.columns else 0,
            'female_count': len(df[df['sex'] == 'Female']) if 'sex' in df.columns else 0,
            'avg_confidence': round(df['confidence'].mean(), 1) if 'confidence' in df.columns else 0
        }

# Initialize storage
storage = StaticStorage(PREDICTIONS_FILE)

# Heart Disease Predictor Class (unchanged)
class HeartDiseasePredictor:
    def __init__(self):
        self.feature_info = {
            'Age': {'type': 'int', 'range': (29, 77), 'description': 'Age in years'},
            'Sex': {'type': 'categorical', 'options': ['Male', 'Female']},
            'ChestPainType': {'type': 'categorical', 'options': 
                             ['Typical Angina', 'Atypical Angina', 
                              'Non-anginal Pain', 'Asymptomatic']},
            'Cholesterol': {'type': 'int', 'range': (126, 564), 
                           'description': 'Serum cholesterol in mg/dl'},
            'FastingBS': {'type': 'binary', 'options': ['≤120 mg/dl', '>120 mg/dl']},
            'MaxHR': {'type': 'int', 'range': (71, 202), 
                     'description': 'Maximum heart rate achieved'},
            'ExerciseAngina': {'type': 'binary', 'options': ['No', 'Yes']},
            'Oldpeak': {'type': 'float', 'range': (0.0, 6.2), 
                       'description': 'ST depression induced by exercise'},
            'ST_Slope': {'type': 'categorical', 'options': 
                        ['Upsloping', 'Flat', 'Downsloping']}
        }
        
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load the trained SVM model"""
        try:
            model_path = 'ui/models/heart_disease_svm_model.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                print("Model loaded successfully")
            else:
                print("Model file not found. Using rule-based prediction.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        # Convert categorical variables
        sex_mapping = {'Male': 1, 'Female': 0}
        cp_mapping = {'Typical Angina': 1, 'Atypical Angina': 2, 
                      'Non-anginal Pain': 3, 'Asymptomatic': 4}
        fbs_mapping = {'≤120 mg/dl': 0, '>120 mg/dl': 1}
        exang_mapping = {'No': 0, 'Yes': 1}
        slope_mapping = {'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}
        
        processed = [
            int(input_data['age']),
            sex_mapping[input_data['sex']],
            cp_mapping[input_data['chest_pain_type']],
            int(input_data['cholesterol']),
            fbs_mapping[input_data['fasting_bs']],
            int(input_data['max_hr']),
            exang_mapping[input_data['exercise_angina']],
            float(input_data['oldpeak']),
            slope_mapping[input_data['st_slope']]
        ]
        
        return np.array(processed).reshape(1, -1)
    
    def predict(self, input_data):
        """Make prediction using SVM or rule-based method"""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            if self.model is not None and self.scaler is not None:
                # Scale the data
                scaled_data = self.scaler.transform(processed_data)
                
                # Make prediction
                prediction = self.model.predict(scaled_data)[0]
                probabilities = self.model.predict_proba(scaled_data)[0]
                
                # For binary classification
                if prediction == 0:
                    result = "No Heart Disease"
                    confidence = probabilities[0]
                    risk_level = "Low"
                else:
                    result = "Heart Disease Detected"
                    confidence = probabilities[1]
                    risk_level = "High" if confidence > 0.7 else "Moderate"
            else:
                # Rule-based prediction (fallback)
                result, confidence, risk_level = self.rule_based_prediction(input_data)
            
            return {
                'result': result,
                'confidence': round(float(confidence) * 100, 2),
                'risk_level': risk_level,
                'details': self.get_risk_factors(input_data)
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'result': "Error in prediction",
                'confidence': 0,
                'risk_level': "Unknown",
                'details': []
            }
    
    def rule_based_prediction(self, input_data):
        """Rule-based prediction when model is not available"""
        risk_score = 0
        factors = []
        
        # Age risk
        age = int(input_data['age'])
        if age > 45:
            risk_score += 1
            factors.append(f"Age ({age}) > 45 years")
        
        # Cholesterol risk
        cholesterol = int(input_data['cholesterol'])
        if cholesterol > 200:
            risk_score += 1
            factors.append(f"Cholesterol ({cholesterol}) > 200 mg/dl")
        
        # Max HR risk
        max_hr = int(input_data['max_hr'])
        if max_hr < 120:
            risk_score += 1
            factors.append(f"Max Heart Rate ({max_hr}) < 120 bpm")
        
        # Oldpeak risk
        oldpeak = float(input_data['oldpeak'])
        if oldpeak > 1.0:
            risk_score += 1
            factors.append(f"ST Depression ({oldpeak}) > 1.0")
        
        # Chest pain type
        if input_data['chest_pain_type'] == 'Asymptomatic':
            risk_score += 1
            factors.append("Asymptomatic chest pain")
        
        # Exercise angina
        if input_data['exercise_angina'] == 'Yes':
            risk_score += 1
            factors.append("Exercise-induced angina present")
        
        # Calculate probability
        probability = min(0.95, risk_score * 0.25)
        
        if probability > 0.5:
            result = "Heart Disease Detected"
            risk_level = "High" if probability > 0.7 else "Moderate"
        else:
            result = "No Heart Disease"
            risk_level = "Low"
        
        return result, probability, risk_level
    
    def get_risk_factors(self, input_data):
        """Identify key risk factors from input data"""
        factors = []
        
        # High cholesterol
        if int(input_data['cholesterol']) > 240:
            factors.append({
                'factor': 'High Cholesterol',
                'level': 'High',
                'description': 'Cholesterol level above 240 mg/dl significantly increases heart disease risk'
            })
        
        # Low max heart rate
        if int(input_data['max_hr']) < 120:
            factors.append({
                'factor': 'Low Maximum Heart Rate',
                'level': 'Moderate',
                'description': 'Maximum heart rate below 120 bpm may indicate poor cardiovascular fitness'
            })
        
        # High ST depression
        if float(input_data['oldpeak']) > 2.0:
            factors.append({
                'factor': 'High ST Depression',
                'level': 'High',
                'description': 'ST depression > 2.0 indicates potential myocardial ischemia'
            })
        
        # Asymptomatic chest pain
        if input_data['chest_pain_type'] == 'Asymptomatic':
            factors.append({
                'factor': 'Asymptomatic Chest Pain',
                'level': 'High',
                'description': 'Asymptomatic presentation can indicate silent ischemia'
            })
        
        return factors

# Initialize predictor
predictor = HeartDiseasePredictor()

# Routes
@app.route('/')
def index():
    """Home page"""
    stats = storage.get_stats()
    recent_predictions = storage.get_predictions(limit=5)
    
    return render_template('index.html', 
                         total_predictions=stats['total_predictions'],
                         high_risk=stats['high_risk'],
                         recent_predictions=recent_predictions)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'POST':
        # Get form data
        patient_data = {
            'patient_name': request.form.get('patient_name'),
            'age': request.form.get('age'),
            'sex': request.form.get('sex'),
            'chest_pain_type': request.form.get('chest_pain_type'),
            'cholesterol': request.form.get('cholesterol'),
            'fasting_bs': request.form.get('fasting_bs'),
            'max_hr': request.form.get('max_hr'),
            'exercise_angina': request.form.get('exercise_angina'),
            'oldpeak': request.form.get('oldpeak'),
            'st_slope': request.form.get('st_slope')
        }
        
        # Validate required fields
        required_fields = ['patient_name', 'age', 'sex', 'chest_pain_type', 
                          'cholesterol', 'fasting_bs', 'max_hr', 
                          'exercise_angina', 'oldpeak', 'st_slope']
        
        for field in required_fields:
            if not patient_data[field]:
                return jsonify({'error': f'{field.replace("_", " ").title()} is required'})
        
        # Make prediction
        result = predictor.predict(patient_data)
        
        # Prepare data for storage
        prediction_to_save = {
            'patient_name': patient_data['patient_name'],
            'age': int(patient_data['age']),
            'sex': patient_data['sex'],
            'chest_pain_type': patient_data['chest_pain_type'],
            'cholesterol': int(patient_data['cholesterol']),
            'fasting_bs': patient_data['fasting_bs'],
            'max_hr': int(patient_data['max_hr']),
            'exercise_angina': patient_data['exercise_angina'],
            'oldpeak': float(patient_data['oldpeak']),
            'st_slope': patient_data['st_slope'],
            'prediction_result': result['result'],
            'confidence': result['confidence'],
            'risk_level': result['risk_level']
        }
        
        # Save to static storage
        prediction_id = storage.save_prediction(prediction_to_save)
        
        # Store in session for results page
        session['prediction_result'] = result
        session['patient_data'] = patient_data
        session['prediction_id'] = prediction_id
        
        return redirect(url_for('results'))
    
    return render_template('predict.html', feature_info=predictor.feature_info)

@app.route('/results')
def results():
    """Results page"""
    result = session.get('prediction_result')
    patient_data = session.get('patient_data')
    prediction_id = session.get('prediction_id')
    
    if not result or not patient_data:
        return redirect(url_for('predict'))
    
    return render_template('results.html', 
                         result=result, 
                         patient_data=patient_data,
                         prediction_id=prediction_id)

@app.route('/dashboard')
def dashboard():
    """Dashboard with analytics"""
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
    if len(df) > 0:
        # Age distribution
        age_fig = px.histogram(df, x='age', nbins=20, 
                              title='Age Distribution of Patients',
                              labels={'age': 'Age', 'count': 'Number of Patients'},
                              color_discrete_sequence=['#6366f1'])
        age_chart = json.dumps(age_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Risk level distribution
        if 'risk_level' in df.columns:
            risk_counts = df['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['risk_level', 'count']
            risk_fig = px.pie(risk_counts, values='count', names='risk_level',
                             title='Risk Level Distribution',
                             color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444'])
            risk_chart = json.dumps(risk_fig, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            risk_chart = None
        
        # Daily trend
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            daily_trend = df.groupby(df['timestamp'].dt.date).size().reset_index()
            daily_trend.columns = ['date', 'count']
            trend_fig = px.line(daily_trend, x='date', y='count',
                               title='Daily Prediction Trends',
                               labels={'date': 'Date', 'count': 'Number of Predictions'},
                               color_discrete_sequence=['#8b5cf6'])
            trend_chart = json.dumps(trend_fig, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            trend_chart = None
    else:
        age_chart = None
        risk_chart = None
        trend_chart = None
    
    # Get statistics
    stats = storage.get_stats()
    
    return render_template('dashboard.html',
                         age_chart=age_chart,
                         risk_chart=risk_chart,
                         trend_chart=trend_chart,
                         stats=stats)

@app.route('/api/predictions')
def get_predictions():
    """API endpoint to get prediction history"""
    predictions = storage.get_predictions(limit=50)
    
    # Format timestamps if they exist
    formatted_predictions = []
    for pred in predictions:
        formatted_pred = pred.copy()
        if 'timestamp' in formatted_pred:
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

@app.route('/api/delete/<int:prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    """Delete a prediction record"""
    success = storage.delete_prediction(prediction_id)
    if success:
        return jsonify({'success': True})
    return jsonify({'error': 'Prediction not found'}), 404

@app.route('/api/export/<format_type>')
def export_data(format_type):
    """Export predictions in specified format"""
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)