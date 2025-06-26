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

#model, preprocessor, metadata = load_model()

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

Live tail
GMT+2

Menu
Collecting fonttools>=4.22.0
  Downloading fonttools-4.58.4-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 24.2 MB/s eta 0:00:00
Collecting kiwisolver>=1.0.1
  Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 9.8 MB/s eta 0:00:00
Collecting cycler>=0.10
  Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
Collecting contourpy>=1.0.1
  Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (325 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 325.0/325.0 kB 981.2 kB/s eta 0:00:00
Collecting geographiclib<3,>=1.52
  Downloading geographiclib-2.0-py3-none-any.whl (40 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.3/40.3 kB 166.4 kB/s eta 0:00:00
Collecting cloudpickle
  Downloading cloudpickle-3.1.1-py3-none-any.whl (20 kB)
Collecting slicer==0.0.7
  Downloading slicer-0.0.7-py3-none-any.whl (14 kB)
Collecting tqdm>=4.27.0
  Downloading tqdm-4.67.1-py3-none-any.whl (78 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.5/78.5 kB 11.7 MB/s eta 0:00:00
Collecting numba
  Downloading numba-0.61.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.8/3.8 MB 100.6 MB/s eta 0:00:00
Collecting tenacity>=6.2.0
  Downloading tenacity-9.1.2-py3-none-any.whl (28 kB)
Collecting MarkupSafe>=2.0
  Downloading MarkupSafe-3.0.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (20 kB)
Collecting six>=1.5
  Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
Collecting llvmlite<0.45,>=0.44.0dev0
  Downloading llvmlite-0.44.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (42.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 42.4/42.4 MB 47.2 MB/s eta 0:00:00
Collecting numba
  Downloading numba-0.61.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.8/3.8 MB 137.2 MB/s eta 0:00:00
  Downloading numba-0.60.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.7/3.7 MB 126.5 MB/s eta 0:00:00
Collecting llvmlite<0.44,>=0.43.0dev0
  Downloading llvmlite-0.43.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (43.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 43.9/43.9 MB 45.3 MB/s eta 0:00:00
Installing collected packages: pytz, tqdm, threadpoolctl, tenacity, slicer, six, pyparsing, pillow, packaging, numpy, MarkupSafe, llvmlite, kiwisolver, joblib, itsdangerous, geographiclib, fonttools, cycler, cloudpickle, click, blinker, Werkzeug, scipy, python-dateutil, plotly, numba, Jinja2, gunicorn, geopy, contourpy, scikit-learn, pandas, matplotlib, Flask, shap, seaborn
Successfully installed Flask-2.3.3 Jinja2-3.1.6 MarkupSafe-3.0.2 Werkzeug-3.1.3 blinker-1.9.0 click-8.2.1 cloudpickle-3.1.1 contourpy-1.3.2 cycler-0.12.1 fonttools-4.58.4 geographiclib-2.0 geopy-2.3.0 gunicorn-21.2.0 itsdangerous-2.2.0 joblib-1.2.0 kiwisolver-1.4.8 llvmlite-0.43.0 matplotlib-3.7.2 numba-0.60.0 numpy-1.23.5 packaging-25.0 pandas-1.5.3 pillow-11.2.1 plotly-5.17.0 pyparsing-3.0.9 python-dateutil-2.9.0.post0 pytz-2025.2 scikit-learn-1.2.2 scipy-1.15.3 seaborn-0.12.2 shap-0.43.0 six-1.17.0 slicer-0.0.7 tenacity-9.1.2 threadpoolctl-3.6.0 tqdm-4.67.1
[notice] A new release of pip is available: 23.0.1 -> 25.1.1
[notice] To update, run: pip install --upgrade pip
==> Uploading build...
==> Uploaded in 15.1s. Compression took 5.9s
==> Build successful 🎉
==> Deploying...
==> Running 'gunicorn app:app'
==> No open ports detected, continuing to scan...
==> Docs on specifying a port: https://render.com/docs/web-services#port-binding
Matplotlib is building the font cache; this may take a moment.
Traceback (most recent call last):
  File "/opt/render/project/src/.venv/bin/gunicorn", line 8, in <module>
    sys.exit(run())
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/app/wsgiapp.py", line 67, in run
    WSGIApplication("%(prog)s [OPTIONS] [APP_MODULE]").run()
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/app/base.py", line 236, in run
    super().run()
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/app/base.py", line 72, in run
    Arbiter(self).run()
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/arbiter.py", line 58, in __init__
    self.setup(app)
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/arbiter.py", line 118, in setup
    self.app.wsgi()
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/app/base.py", line 67, in wsgi
    self.callable = self.load()
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/app/wsgiapp.py", line 58, in load
    return self.load_wsgiapp()
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/app/wsgiapp.py", line 48, in load_wsgiapp
    return util.import_app(self.app_uri)
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/util.py", line 371, in import_app
    mod = importlib.import_module(module)
  File "/opt/render/project/python/Python-3.10.11/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/opt/render/project/src/app.py", line 26, in <module>
    model, preprocessor, metadata = load_model()
  File "/opt/render/project/src/app.py", line 17, in load_model
    model = joblib.load('models/best_model_gradient_boosting.joblib')
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/joblib/numpy_pickle.py", line 658, in load
    obj = _unpickle(fobj, filename, mmap_mode)
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/joblib/numpy_pickle.py", line 577, in _unpickle
    obj = unpickler.load()
  File "/opt/render/project/python/Python-3.10.11/lib/python3.10/pickle.py", line 1213, in load
    dispatch[key[0]](self)
  File "/opt/render/project/python/Python-3.10.11/lib/python3.10/pickle.py", line 1538, in load_stack_global
    self.append(self.find_class(module, name))
  File "/opt/render/project/python/Python-3.10.11/lib/python3.10/pickle.py", line 1580, in find_class
    __import__(module, level=0)
ModuleNotFoundError: No module named '_loss'

Live tail
GMT+2

Menu
Collecting fonttools>=4.22.0
  Downloading fonttools-4.58.4-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 24.2 MB/s eta 0:00:00
Collecting kiwisolver>=1.0.1
  Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 9.8 MB/s eta 0:00:00
Collecting cycler>=0.10
  Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
Collecting contourpy>=1.0.1
  Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (325 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 325.0/325.0 kB 981.2 kB/s eta 0:00:00
Collecting geographiclib<3,>=1.52
  Downloading geographiclib-2.0-py3-none-any.whl (40 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.3/40.3 kB 166.4 kB/s eta 0:00:00
Collecting cloudpickle
  Downloading cloudpickle-3.1.1-py3-none-any.whl (20 kB)
Collecting slicer==0.0.7
  Downloading slicer-0.0.7-py3-none-any.whl (14 kB)
Collecting tqdm>=4.27.0
  Downloading tqdm-4.67.1-py3-none-any.whl (78 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.5/78.5 kB 11.7 MB/s eta 0:00:00
Collecting numba
  Downloading numba-0.61.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.8/3.8 MB 100.6 MB/s eta 0:00:00
Collecting tenacity>=6.2.0
  Downloading tenacity-9.1.2-py3-none-any.whl (28 kB)
Collecting MarkupSafe>=2.0
  Downloading MarkupSafe-3.0.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (20 kB)
Collecting six>=1.5
  Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
Collecting llvmlite<0.45,>=0.44.0dev0
  Downloading llvmlite-0.44.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (42.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 42.4/42.4 MB 47.2 MB/s eta 0:00:00
Collecting numba
  Downloading numba-0.61.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.8/3.8 MB 137.2 MB/s eta 0:00:00
  Downloading numba-0.60.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.7/3.7 MB 126.5 MB/s eta 0:00:00
Collecting llvmlite<0.44,>=0.43.0dev0
  Downloading llvmlite-0.43.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (43.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 43.9/43.9 MB 45.3 MB/s eta 0:00:00
Installing collected packages: pytz, tqdm, threadpoolctl, tenacity, slicer, six, pyparsing, pillow, packaging, numpy, MarkupSafe, llvmlite, kiwisolver, joblib, itsdangerous, geographiclib, fonttools, cycler, cloudpickle, click, blinker, Werkzeug, scipy, python-dateutil, plotly, numba, Jinja2, gunicorn, geopy, contourpy, scikit-learn, pandas, matplotlib, Flask, shap, seaborn
Successfully installed Flask-2.3.3 Jinja2-3.1.6 MarkupSafe-3.0.2 Werkzeug-3.1.3 blinker-1.9.0 click-8.2.1 cloudpickle-3.1.1 contourpy-1.3.2 cycler-0.12.1 fonttools-4.58.4 geographiclib-2.0 geopy-2.3.0 gunicorn-21.2.0 itsdangerous-2.2.0 joblib-1.2.0 kiwisolver-1.4.8 llvmlite-0.43.0 matplotlib-3.7.2 numba-0.60.0 numpy-1.23.5 packaging-25.0 pandas-1.5.3 pillow-11.2.1 plotly-5.17.0 pyparsing-3.0.9 python-dateutil-2.9.0.post0 pytz-2025.2 scikit-learn-1.2.2 scipy-1.15.3 seaborn-0.12.2 shap-0.43.0 six-1.17.0 slicer-0.0.7 tenacity-9.1.2 threadpoolctl-3.6.0 tqdm-4.67.1
[notice] A new release of pip is available: 23.0.1 -> 25.1.1
[notice] To update, run: pip install --upgrade pip
==> Uploading build...
==> Uploaded in 15.1s. Compression took 5.9s
==> Build successful 🎉
==> Deploying...
==> Running 'gunicorn app:app'
==> No open ports detected, continuing to scan...
==> Docs on specifying a port: https://render.com/docs/web-services#port-binding
Matplotlib is building the font cache; this may take a moment.
Traceback (most recent call last):
  File "/opt/render/project/src/.venv/bin/gunicorn", line 8, in <module>
    sys.exit(run())
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/app/wsgiapp.py", line 67, in run
    WSGIApplication("%(prog)s [OPTIONS] [APP_MODULE]").run()
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/app/base.py", line 236, in run
    super().run()
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/app/base.py", line 72, in run
    Arbiter(self).run()
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/arbiter.py", line 58, in __init__
    self.setup(app)
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/arbiter.py", line 118, in setup
    self.app.wsgi()
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/app/base.py", line 67, in wsgi
    self.callable = self.load()
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/app/wsgiapp.py", line 58, in load
    return self.load_wsgiapp()
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/app/wsgiapp.py", line 48, in load_wsgiapp
    return util.import_app(self.app_uri)
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/gunicorn/util.py", line 371, in import_app
    mod = importlib.import_module(module)
  File "/opt/render/project/python/Python-3.10.11/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/opt/render/project/src/app.py", line 26, in <module>
    model, preprocessor, metadata = load_model()
  File "/opt/render/project/src/app.py", line 17, in load_model
    model = joblib.load('models/best_model_gradient_boosting.joblib')
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/joblib/numpy_pickle.py", line 658, in load
    obj = _unpickle(fobj, filename, mmap_mode)
  File "/opt/render/project/src/.venv/lib/python3.10/site-packages/joblib/numpy_pickle.py", line 577, in _unpickle
    obj = unpickler.load()
  File "/opt/render/project/python/Python-3.10.11/lib/python3.10/pickle.py", line 1213, in load
    dispatch[key[0]](self)
  File "/opt/render/project/python/Python-3.10.11/lib/python3.10/pickle.py", line 1538, in load_stack_global
    self.append(self.find_class(module, name))
  File "/opt/render/project/python/Python-3.10.11/lib/python3.10/pickle.py", line 1580, in find_class
    __import__(module, level=0)
ModuleNotFoundError: No module named '_loss'
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