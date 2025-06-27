from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from model_explainer import ModelExplainer
import random

app = Flask(__name__)

# Load the saved model, preprocessor, and metadata
def load_model():
    try:
        model = joblib.load('models/best_model_gradient_boosting.joblib')
        preprocessor = joblib.load('models/preprocessor.joblib')
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, preprocessor, metadata
    except FileNotFoundError:
        print("Model files not found. Please run main.py first to train and save the model.")
        return None, None, None

# Initialize model variables (commented out due to compatibility issues)
#model, preprocessor, metadata = None, None, None
model, preprocessor, metadata = load_model()

# Initialize model explainer
try:
    explainer = ModelExplainer()
except Exception as e:
    print(f"Could not initialize model explainer: {e}")
    explainer = None

# Define known locations and their coordinates from your dataset
# In a real application, you would load this from a file or database
known_locations = {
    'CBD': (-17.8252, 31.0335),
    'Avenues': (-17.8189, 31.0494),
    'Eastlea': (-17.8318, 31.0833),
    'Avondale': (-17.7928, 31.0250),
    'Borrowdale': (-17.7562, 31.0772),
    # Add other locations from your CSV here
}

def find_nearest_location(lat, lon):
    """Finds the nearest known location from the user's coordinates."""
    user_location = (lat, lon)
    nearest_location = None
    min_distance = float('inf')
    
    for loc, coords in known_locations.items():
        distance = geodesic(user_location, coords).km
        if distance < min_distance:
            min_distance = distance
            nearest_location = loc
            
    return nearest_location

@app.route('/')
def home():
    """Introductory home page."""
    return render_template('intro.html')

@app.route('/options')
def options():
    """Page with user options (buy, sell, predict)."""
    return render_template('options.html')

@app.route('/predict-form')
def predict_form():
    """Serves the property valuation form."""
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle property valuation prediction."""
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Get location from coordinates
        latitude = float(form_data.pop('latitude'))
        longitude = float(form_data.pop('longitude'))
        detected_location = find_nearest_location(latitude, longitude)
        
        if not detected_location:
            return jsonify({'success': False, 'error': 'Could not determine a known location from your coordinates.'})

        # Prepare data for prediction
        data = {
            'Property Type': form_data['property_type'],
            'Location': detected_location,
            'Area': float(form_data['area']),
            'Number of structures': int(form_data['structures']),
            'Land Rate': float(form_data['land_rate']),
            'Number of rooms': int(form_data['rooms']),
            'Swimming Pool': form_data['swimming_pool'],
            'Boundary': form_data['boundary'],
            'Age Category': form_data['age_category'],
            'Proximity to schools(km)': float(form_data['schools']),
            'Proximity to healthcare(km)': float(form_data['healthcare']),
            'Proximity to malls(km)': float(form_data['malls']),
            'Proximity to highway(km)': float(form_data['highway'])
        }
        
        # Create DataFrame and predict
        df = pd.DataFrame([data])
        
        # Ensure model is loaded
        if model is None or preprocessor is None:
            return jsonify({
                'success': False, 
                'error': 'Model failed to load. Please try again later.'
            })
        
        # Process data and make prediction
        X_processed = preprocessor.transform(df)
        prediction = model.predict(X_processed)[0]
        
        # Get model info
        model_info = "Gradient Boosting Model"
        if metadata is not None and 'performance' in metadata:
            model_info = f"Gradient Boosting Model - MAE: ${metadata['performance']['MAE']:,.2f}"
        
        return jsonify({
            'success': True,
            'prediction': f"${prediction:,.2f}",
            'location': detected_location,
            'confidence': 'High (R² = 0.93)',
            'model_info': model_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/dashboard')
def dashboard():
    """Dashboard page for property value trends and patterns."""
    return render_template('dashboard.html')

@app.route('/spatial-analysis')
def spatial_analysis():
    """Spatial analysis page for property value distribution."""
    return render_template('spatial_analysis.html')

@app.route('/api/property-data')
def get_property_data():
    """API endpoint to get property data for visualizations."""
    try:
        df = pd.read_csv('PROPERTYVALUATIONS.csv')
        
        # Add status field (sold/available) - for demo purposes, randomly assign
        df['Status'] = df.apply(lambda x: random.choice(['Sold', 'Available']), axis=1)
        
        return jsonify({
            'success': True,
            'data': df.to_dict('records'),
            'total_properties': len(df),
            'avg_value': df['Market Value'].mean(),
            'locations': df['Location'].unique().tolist()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/buy')
def buy():
    """Buy page for property listings."""
    return render_template('buy.html')

if __name__ == '__main__':
    app.run(debug=True)

