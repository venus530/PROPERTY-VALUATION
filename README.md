# Harare Property Valuation System

A comprehensive machine learning-based property valuation system for Harare, Zimbabwe, developed as part of a dissertation project. This system provides accurate property value predictions using spatial and non-spatial attributes, along with interactive visualizations and analysis tools.

## 🎯 Project Objectives

This project addresses four key dissertation objectives:

1. **Web-based Application Development**: Create a user-friendly web interface for property valuation
2. **Predictive Model with Spatial Attributes**: Develop ML models incorporating both spatial and non-spatial property features
3. **Interactive Dashboard**: Build comprehensive visualizations for property market trends
4. **Spatial Distribution Analysis**: Analyze and visualize property value distribution across Harare

## 🏗️ Architecture

### Backend
- **Framework**: Flask (Python)
- **Machine Learning**: Scikit-learn with Gradient Boosting, Random Forest, SVM, and Linear Regression
- **Data Processing**: Pandas for data manipulation and preprocessing
- **Model Persistence**: Joblib for model serialization

### Frontend
- **UI Framework**: Bootstrap 5 for responsive design
- **Charts**: Chart.js for interactive data visualizations
- **Maps**: Leaflet.js for spatial analysis and mapping
- **JavaScript**: jQuery for AJAX requests and DOM manipulation

### Models
- **Best Model**: Gradient Boosting Regressor (R² = 0.93, MAE ≈ $9,600)
- **Features**: 8 numerical + 5 categorical attributes
- **Preprocessing**: StandardScaler for numerical, OneHotEncoder for categorical features

## 🚀 Features

### Core Functionality
- **Property Value Prediction**: ML-powered valuation with 93% accuracy
- **Location Detection**: Automatic GPS-based location detection
- **Multiple Use Cases**: Predict & Sell, Buy, and Predict Only modes
- **Real-time Analysis**: Live data processing and visualization

### Interactive Dashboard
- **Market Statistics**: Total properties, average values, price ranges
- **Location Analysis**: Property values by area with interactive charts
- **Property Type Distribution**: Visual breakdown of property types
- **Data Table**: Comprehensive property data overview

### Spatial Analysis
- **Interactive Map**: Harare property value distribution
- **Value Hotspots**: Identification of premium and affordable areas
- **Location Insights**: Detailed analysis by neighborhood
- **Spatial Statistics**: Geographic market analysis

## 📊 Model Performance

| Model | R² Score | MAE | MSE |
|-------|----------|-----|-----|
| Gradient Boosting | 0.93 | $9,600 | 1.2e8 |
| Random Forest | 0.91 | $10,200 | 1.4e8 |
| Support Vector Machine | 0.89 | $11,500 | 1.7e8 |
| Linear Regression | 0.85 | $13,100 | 2.1e8 |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip package manager

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/harare-property-valuation.git
   cd harare-property-valuation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   - Open your browser and go to `http://127.0.0.1:5000`
   - The application will be available on your local machine

### Production Deployment

#### Option 1: Render (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Render**
   - Go to [render.com](https://render.com) and create an account
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository
   - Configure the service:
     - **Name**: harare-property-valuation
     - **Environment**: Python
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
   - Click "Create Web Service"

3. **Automatic Deployment**
   - Render will automatically deploy your application
   - Each push to the main branch triggers a new deployment
   - Your app will be available at `https://your-app-name.onrender.com`

#### Option 2: Heroku

1. **Install Heroku CLI**
   ```bash
   # Download and install from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Deploy to Heroku**
   ```bash
   heroku create your-app-name
   git push heroku main
   heroku open
   ```

#### Option 3: PythonAnywhere

1. **Create PythonAnywhere account**
   - Go to [pythonanywhere.com](https://pythonanywhere.com)
   - Create a free account

2. **Upload and configure**
   - Upload your project files
   - Configure WSGI file
   - Set up virtual environment
   - Install requirements

## 📁 Project Structure

```
property_valuation_project/
├── app.py                          # Flask application
├── main.py                         # ML model training script
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render deployment config
├── Procfile                        # Heroku deployment config
├── PROPERTYVALUATIONS.csv          # Dataset
├── models/                         # Trained models
│   ├── best_model_gradient_boosting.joblib
│   ├── preprocessor.joblib
│   └── model_metadata.json
├── templates/                      # HTML templates
│   ├── index.html                  # Home page
│   ├── intro.html                  # Introduction
│   ├── options.html                # Service options
│   ├── predict.html                # Prediction form
│   ├── dashboard.html              # Interactive dashboard
│   └── spatial_analysis.html       # Spatial analysis
└── static/                         # Static assets (if any)
```

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: Set to `production` for deployment
- `FLASK_DEBUG`: Set to `False` in production

### Model Configuration
- Models are automatically saved in the `models/` directory
- Preprocessor and metadata are preserved for consistent predictions
- Model performance metrics are logged and visualized

## 📈 API Endpoints

### Web Routes
- `GET /`: Home page
- `GET /intro`: Introduction page
- `GET /options`: Service selection
- `GET /predict-form`: Prediction form
- `GET /dashboard`: Interactive dashboard
- `GET /spatial-analysis`: Spatial analysis

### API Endpoints
- `POST /predict`: Property value prediction
- `GET /api/property-data`: Property data for visualizations

## 🧪 Testing

### Model Testing
```bash
python main.py
```
This will:
- Load and preprocess the dataset
- Train multiple ML models
- Perform hyperparameter tuning
- Generate performance visualizations
- Save the best model

### Application Testing
```bash
python app.py
```
Then visit:
- `http://127.0.0.1:5000` - Test home page
- `http://127.0.0.1:5000/dashboard` - Test dashboard
- `http://127.0.0.1:5000/spatial-analysis` - Test spatial analysis

## 📊 Data Sources

The system uses a comprehensive dataset of Harare properties including:
- **Spatial Attributes**: Location, proximity to amenities
- **Physical Attributes**: Area, rooms, property type
- **Market Data**: Current market values, land rates
- **Infrastructure**: Swimming pools, boundary types

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is developed for academic research purposes. Please ensure proper attribution when using this code.

## 👨‍🎓 Dissertation Information

**Title**: Machine Learning-Based Property Valuation System for Harare, Zimbabwe
**Author**: [Your Name]
**Institution**: [Your University]
**Year**: 2024

### Research Contributions
- Novel application of ML in Zimbabwean real estate
- Integration of spatial and non-spatial attributes
- Development of accessible web-based valuation tools
- Comprehensive market analysis framework

## 📞 Support

For questions or support:
- Create an issue on GitHub
- Contact: [your-email@university.edu]

---

**Note**: This system is designed for educational and research purposes. For commercial property valuations, please consult with licensed real estate professionals. 