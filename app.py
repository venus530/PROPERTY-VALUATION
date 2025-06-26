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
        X_processed = preprocessor.transform(df)
        prediction = model.predict(X_processed)[0]
        
        return jsonify({
            'success': True,
            'prediction': f"${prediction:,.2f}",
            'location': detected_location,
            'confidence': 'High (R² = 0.93)',
            'model_info': f"Gradient Boosting Model - MAE: ${metadata['performance']['MAE']:,.2f}"
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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
    """Page to list available properties for sale (demo: all from CSV)."""
    try:
        df = pd.read_csv('PROPERTYVALUATIONS.csv')
        # Show a sample of 20 for demo, or all if you prefer
        properties = df.head(20).to_dict('records')
        return render_template('buy.html', properties=properties)
    except Exception as e:
        return f"Error loading properties: {e}", 500

@app.route('/api/available-properties')
def api_available_properties():
    """API endpoint for available properties (for map/listing)."""
    try:
        df = pd.read_csv('PROPERTYVALUATIONS.csv')
        return jsonify({'success': True, 'data': df.to_dict('records')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/explain')
def explain():
    """Model explainability page."""
    return render_template('explain.html')

@app.route('/explain-prediction', methods=['POST'])
def explain_prediction():
    """Generate explanation for a property prediction."""
    if explainer is None:
        return jsonify({'success': False, 'error': 'Model explainer not available'})
    
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Get location from coordinates
        latitude = float(form_data.pop('latitude'))
        longitude = float(form_data.pop('longitude'))
        detected_location = find_nearest_location(latitude, longitude)
        
        if not detected_location:
            return jsonify({'success': False, 'error': 'Could not determine a known location from your coordinates.'})

        # Prepare data for explanation
        data = {
            'bedrooms': int(form_data['bedrooms']),
            'bathrooms': int(form_data['bathrooms']),
            'parking_spaces': int(form_data['parking_spaces']),
            'size_sqm': float(form_data['size_sqm']),
            'distance_to_city_center': float(form_data['distance_to_city_center']),
            'distance_to_nearest_school': float(form_data['distance_to_nearest_school']),
            'distance_to_nearest_hospital': float(form_data['distance_to_nearest_hospital']),
            'crime_rate': float(form_data['crime_rate']),
            'area': detected_location,
            'property_type': form_data['property_type'],
            'condition': form_data['condition'],
            'furnished': form_data['furnished'],
            'garden': form_data['garden'],
            'pool': form_data['pool']
        }
        
        # Generate explanation
        explanation = explainer.explain_prediction(data)
        
        if "error" in explanation:
            return jsonify({'success': False, 'error': explanation['error']})
        
        # Create plots
        feature_plot = explainer.create_feature_importance_plot(explanation)
        waterfall_plot = explainer.create_waterfall_plot(explanation)
        
        return jsonify({
            'success': True,
            'explanation': explanation,
            'feature_plot': feature_plot,
            'waterfall_plot': waterfall_plot,
            'location': detected_location
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/global-importance')
def global_importance():
    """Show global feature importance."""
    if explainer is None:
        return jsonify({'success': False, 'error': 'Model explainer not available'})
    
    try:
        global_plot = explainer.create_global_importance_plot()
        global_importance = explainer.get_global_feature_importance()
        
        return jsonify({
            'success': True,
            'global_plot': global_plot,
            'global_importance': global_importance
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Get port from environment variable (for Render) or use 5000 for local development
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port) 