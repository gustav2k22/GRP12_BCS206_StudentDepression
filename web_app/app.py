from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingClassifier
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class StudentDepressionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None
        self.model_loaded = False
        self.setup_model()
    
    def setup_model(self):
        """Setup the prediction model with preprocessing"""
        try:
            # Load the actual trained model
            model_path = 'model_components.pkl'
            if os.path.exists(model_path):
                print(f"Loading trained model from {model_path}...")
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler'] 
                    self.label_encoders = model_data['label_encoders']
                    self.feature_columns = model_data['feature_columns']
                    self.target_column = model_data.get('target_column', 'Depression')
                    self.model_loaded = True
                print("✅ Actual trained model loaded successfully!")
                print(f"Model type: {type(self.model).__name__}")
                print(f"Number of features: {len(self.feature_columns)}")
                print(f"Available encoders: {list(self.label_encoders.keys())}")
            else:
                print(f"❌ Model file {model_path} not found!")
                print("Please run train_model.py first to train the model")
                raise FileNotFoundError(f"Model file {model_path} not found")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️  Falling back to rule-based prediction system")
            self.model_loaded = False
    
    def preprocess_input(self, form_data):
        """Preprocess form data to match model input format"""
        try:
            if not self.model_loaded:
                raise ValueError("Model not loaded properly")
            
            # Create a data dictionary matching the model's expected format
            processed_data = {
                'id': 1,  # Dummy ID 
                'Gender': form_data.get('gender', 'Male'),
                'Age': float(form_data.get('age', 25)),
                'City': form_data.get('city', 'Mumbai'),
                'Profession': form_data.get('profession', 'Student'),
                'Academic Pressure': float(form_data.get('academicPressure', 2.5)),
                'Work Pressure': float(form_data.get('workPressure', 0)),
                'CGPA': float(form_data.get('cgpa', 7.0)),
                'Study Satisfaction': float(form_data.get('studySatisfaction', 2.5)),
                'Job Satisfaction': float(form_data.get('jobSatisfaction', 0)),
                'Sleep Duration': form_data.get('sleepDuration', '7-8 hours'),
                'Dietary Habits': form_data.get('dietaryHabits', 'Moderate'),
                'Degree': form_data.get('degree', 'B.Tech'),
                'Have you ever had suicidal thoughts ?': form_data.get('suicidalThoughts', 'No'),
                'Work/Study Hours': float(form_data.get('studyHours', 8)),
                'Financial Stress': str(form_data.get('financialStress', '3.0')),  # Keep as string to match training
                'Family History of Mental Illness': form_data.get('familyHistory', 'No')
            }
            
            # Convert to DataFrame
            df = pd.DataFrame([processed_data])
            
            # Ensure all expected columns are present and in correct order
            for col in self.feature_columns:
                if col not in df.columns:
                    # Add missing columns with default values
                    if col == 'id':
                        df[col] = 1
                    else:
                        df[col] = 0
            
            # Reorder columns to match training
            df = df[self.feature_columns]
            
            # Encode categorical variables using the trained encoders
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    try:
                        # Handle unknown categories by using the most common class
                        unknown_mask = ~df[col].astype(str).isin(encoder.classes_)
                        if unknown_mask.any():
                            # Replace unknown values with the first class (most common in training)
                            df.loc[unknown_mask, col] = encoder.classes_[0]
                        
                        df[col] = encoder.transform(df[col].astype(str))
                    except Exception as e:
                        print(f"Warning: Error encoding {col}: {e}")
                        # Set to 0 (first encoded value) as fallback
                        df[col] = 0
            
            return df
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            print(f"Form data received: {form_data}")
            raise
    
    def predict(self, form_data):
        """Make prediction based on form data using ML model or fallback to rule-based"""
        try:
            if not self.model_loaded:
                print("🔄 Using rule-based prediction system")
                return self.rule_based_prediction(form_data)
            
            # Preprocess the input
            processed_df = self.preprocess_input(form_data)
            
            # Scale the features using the trained scaler
            processed_scaled = self.scaler.transform(processed_df)
            
            # Make prediction using the actual trained model
            prediction_proba = self.model.predict_proba(processed_scaled)[0]
            prediction_class = self.model.predict(processed_scaled)[0]
            
            # Get prediction probability (probability of depression = class 1)
            depression_probability = prediction_proba[1] * 100  # Convert to percentage
            
            # Determine risk level based on probability
            if depression_probability >= 70:
                risk_level = "high"
            elif depression_probability >= 40:
                risk_level = "moderate"
            else:
                risk_level = "low"
            
            # Analyze risk factors based on form data
            risk_factors = self.analyze_risk_factors(form_data)
            
            # Calculate confidence based on probability distribution
            confidence = max(prediction_proba) if len(prediction_proba) > 1 else 0.8
            
            prediction_result = {
                'risk_level': risk_level,
                'probability': round(depression_probability, 1),
                'confidence': round(confidence, 2),
                'risk_factors': risk_factors,
                'model_prediction': int(prediction_class),
                'raw_probabilities': {
                    'no_depression': round(prediction_proba[0] * 100, 1),
                    'depression': round(prediction_proba[1] * 100, 1)
                }
            }
            
            print(f"Model prediction: {prediction_result}")
            return prediction_result
            
        except Exception as e:
            print(f"Error in ML prediction, falling back to rule-based: {e}")
            return self.rule_based_prediction(form_data)
    
    def analyze_risk_factors(self, data):
        """Analyze form data to identify specific risk factors"""
        risk_factors = []
        
        # Academic pressure
        academic_pressure = float(data.get('academicPressure', 0))
        if academic_pressure >= 4:
            risk_factors.append("High academic pressure")
        
        # Study satisfaction (inverse relationship)
        study_satisfaction = float(data.get('studySatisfaction', 0))
        if study_satisfaction <= 2:
            risk_factors.append("Low study satisfaction")
        
        # Work pressure
        work_pressure = float(data.get('workPressure', 0))
        if work_pressure >= 3:
            risk_factors.append("High work pressure")
        
        # Financial stress
        financial_stress = int(data.get('financialStress', 3))
        if financial_stress >= 4:
            risk_factors.append("High financial stress")
        
        # Sleep duration
        sleep_duration = data.get('sleepDuration', '')
        if sleep_duration == "Less than 5 hours":
            risk_factors.append("Insufficient sleep")
        
        # Dietary habits
        dietary_habits = data.get('dietaryHabits', '')
        if dietary_habits == "Unhealthy":
            risk_factors.append("Poor dietary habits")
        
        # Suicidal thoughts
        suicidal_thoughts = data.get('suicidalThoughts', '')
        if suicidal_thoughts == "Yes":
            risk_factors.append("History of suicidal thoughts")
        
        # Family history
        family_history = data.get('familyHistory', '')
        if family_history == "Yes":
            risk_factors.append("Family history of mental illness")
        
        # Study hours (too much or too little)
        study_hours = float(data.get('studyHours', 8))
        if study_hours >= 12:
            risk_factors.append("Excessive study hours")
        elif study_hours <= 2:
            risk_factors.append("Very low study engagement")
        
        # CGPA concerns
        cgpa = float(data.get('cgpa', 7.0))
        if cgpa <= 5.0:
            risk_factors.append("Academic performance concerns")
        
        return risk_factors
    
    def rule_based_prediction(self, data):
        """Enhanced rule-based prediction system"""
        risk_score = 0
        risk_factors = []
        confidence_factors = []
        
        # Academic pressure (weight: 0.15)
        academic_pressure = float(data.get('academicPressure', 0))
        if academic_pressure >= 4:
            risk_score += 0.15
            risk_factors.append("High academic pressure")
            confidence_factors.append(0.8)
        elif academic_pressure >= 3:
            risk_score += 0.08
            confidence_factors.append(0.6)
        
        # Study satisfaction (inverse relationship, weight: 0.12)
        study_satisfaction = float(data.get('studySatisfaction', 0))
        if study_satisfaction <= 1.5:
            risk_score += 0.12
            risk_factors.append("Very low study satisfaction")
            confidence_factors.append(0.9)
        elif study_satisfaction <= 2.5:
            risk_score += 0.06
            risk_factors.append("Low study satisfaction")
            confidence_factors.append(0.7)
        
        # Work pressure (weight: 0.10)
        work_pressure = float(data.get('workPressure', 0))
        if work_pressure >= 3:
            risk_score += 0.10
            risk_factors.append("High work pressure")
            confidence_factors.append(0.75)
        
        # Financial stress (weight: 0.18)
        financial_stress = int(data.get('financialStress', 3))
        if financial_stress >= 4:
            risk_score += 0.18
            risk_factors.append("High financial stress")
            confidence_factors.append(0.85)
        elif financial_stress >= 3:
            risk_score += 0.09
            confidence_factors.append(0.6)
        
        # Sleep duration (weight: 0.12)
        sleep_duration = data.get('sleepDuration', '')
        if sleep_duration == "Less than 5 hours":
            risk_score += 0.12
            risk_factors.append("Severely insufficient sleep")
            confidence_factors.append(0.9)
        elif sleep_duration == "5-6 hours":
            risk_score += 0.06
            risk_factors.append("Insufficient sleep")
            confidence_factors.append(0.7)
        
        # Dietary habits (weight: 0.08)
        dietary_habits = data.get('dietaryHabits', '')
        if dietary_habits == "Unhealthy":
            risk_score += 0.08
            risk_factors.append("Poor dietary habits")
            confidence_factors.append(0.6)
        
        # Suicidal thoughts (weight: 0.25)
        suicidal_thoughts = data.get('suicidalThoughts', '')
        if suicidal_thoughts == "Yes":
            risk_score += 0.25
            risk_factors.append("History of suicidal thoughts")
            confidence_factors.append(0.95)
        
        # Family history (weight: 0.10)
        family_history = data.get('familyHistory', '')
        if family_history == "Yes":
            risk_score += 0.10
            risk_factors.append("Family history of mental illness")
            confidence_factors.append(0.8)
        
        # Age factor (younger students often have higher risk)
        age = float(data.get('age', 25))
        if age <= 20:
            risk_score += 0.05
            confidence_factors.append(0.5)
        
        # CGPA factor (very low or high perfectionist pressure)
        cgpa = float(data.get('cgpa', 7.0))
        if cgpa <= 5.0:
            risk_score += 0.08
            risk_factors.append("Academic performance concerns")
            confidence_factors.append(0.7)
        elif cgpa >= 9.5:
            risk_score += 0.03  # Perfectionist pressure
            confidence_factors.append(0.4)
        
        # Study hours (too much or too little)
        study_hours = float(data.get('studyHours', 8))
        if study_hours >= 12:
            risk_score += 0.05
            risk_factors.append("Excessive study hours")
            confidence_factors.append(0.6)
        elif study_hours <= 2:
            risk_score += 0.03
            confidence_factors.append(0.4)
        
        # Calculate confidence
        confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        # Determine risk level and probability
        if risk_score >= 0.6:
            risk_level = "high"
            probability = min(95, 60 + risk_score * 40)
        elif risk_score >= 0.3:
            risk_level = "moderate"  
            probability = 30 + risk_score * 50
        else:
            risk_level = "low"
            probability = max(5, risk_score * 50)
        
        return {
            'risk_level': risk_level,
            'probability': round(probability),
            'confidence': round(confidence, 2),
            'risk_score': round(risk_score, 3),
            'risk_factors': risk_factors,
            'total_factors_checked': 11
        }

# Initialize predictor
predictor = StudentDepressionPredictor()

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Web app files not found. Please ensure index.html is in the same directory.", 404

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('.', filename)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        result = predictor.predict(data)
        
        # Add recommendations based on risk level
        result['recommendations'] = get_recommendations(result['risk_level'])
        result['resources'] = get_mental_health_resources()
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

def get_recommendations(risk_level):
    """Get recommendations based on risk level"""
    base_recommendations = [
        "Maintain a regular sleep schedule (7-9 hours per night)",
        "Engage in regular physical exercise (30 minutes daily)",
        "Practice stress reduction techniques (meditation, deep breathing)",
        "Connect with supportive friends and family regularly",
        "Maintain a balanced diet and stay hydrated"
    ]
    
    moderate_recommendations = [
        "Consider scheduling an appointment with a counselor",
        "Join a support group or peer counseling program", 
        "Develop and maintain a daily routine",
        "Limit alcohol consumption and avoid recreational drugs",
        "Practice mindfulness and relaxation techniques daily",
        "Consider stress management workshops"
    ]
    
    high_recommendations = [
        "Seek immediate professional help from a mental health professional",
        "Contact your doctor for a comprehensive mental health evaluation",
        "Inform a trusted friend or family member about how you're feeling",
        "Consider removing any means of self-harm from your environment",
        "Create a safety plan with professional guidance",
        "Consider intensive outpatient programs if available"
    ]
    
    if risk_level == 'low':
        return base_recommendations[:4]
    elif risk_level == 'moderate':
        return base_recommendations[:2] + moderate_recommendations[:4]
    else:  # high risk
        return high_recommendations

def get_mental_health_resources():
    """Get mental health resources and contact information"""
    return {
        'crisis_lines': {
            'national': '988 (Suicide & Crisis Lifeline)',
            'text': 'Text HOME to 741741 (Crisis Text Line)',
            'international': '+1-800-273-8255'
        },
        'online_resources': [
            {
                'name': 'National Institute of Mental Health',
                'url': 'https://blog.opencounseling.com/suicide-hotlines/',
                'description': 'Find mental health services and resources'
            },
            {
                'name': 'Psychology Today',
                'url': 'https://www.psychologytoday.com/us/therapists',
                'description': 'Find therapists and mental health professionals'
            },
            {
                'name': 'BetterHelp',
                'url': 'https://www.betterhelp.com',
                'description': 'Online therapy and counseling services'
            }
        ],
        'apps': [
            'Headspace (meditation and mindfulness)',
            'Calm (sleep and relaxation)',
            'Sanvello (mood and anxiety tracking)',
            'MindShift (anxiety management)'
        ]
    }

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model_loaded,
        'model_type': type(predictor.model).__name__ if predictor.model else None,
        'features_count': len(predictor.feature_columns) if predictor.feature_columns else 0,
        'encoders_available': list(predictor.label_encoders.keys()) if predictor.label_encoders else [],
        'timestamp': pd.Timestamp.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Student Depression Assessment Web App...")
    print("🏥 Model Status: Loaded and Ready")
    print("🌐 Server will be available at: http://localhost:5000")
    print("📊 Features: Real-time assessment, risk analysis, and recommendations")
    print("\n" + "="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
