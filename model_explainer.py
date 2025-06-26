import shap
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os

class ModelExplainer:
    def __init__(self, model_path='models/best_model_gradient_boosting.joblib', 
                 preprocessor_path='models/preprocessor.joblib'):
        """Initialize the model explainer with trained model and preprocessor."""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.explainer = None
        self.feature_names = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer with background data."""
        try:
            # Load sample data for background
            data = pd.read_csv('PROPERTYVALUATIONS.csv')
            
            # Prepare features (same as in main.py)
            feature_columns = ['bedrooms', 'bathrooms', 'parking_spaces', 'size_sqm', 
                             'distance_to_city_center', 'distance_to_nearest_school', 
                             'distance_to_nearest_hospital', 'crime_rate', 'area', 
                             'property_type', 'condition', 'furnished', 'garden', 'pool']
            
            X = data[feature_columns].copy()
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Transform using preprocessor
            X_transformed = self.preprocessor.transform(X)
            
            # Get feature names after preprocessing
            self.feature_names = self.preprocessor.get_feature_names_out()
            
            # Initialize explainer with background data
            background_data = X_transformed[:100]  # Use first 100 samples as background
            self.explainer = shap.TreeExplainer(self.model, background_data)
            
        except Exception as e:
            print(f"Error initializing explainer: {e}")
            self.explainer = None
    
    def explain_prediction(self, input_data):
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            input_data (dict): Property features
            
        Returns:
            dict: Explanation results with SHAP values and plots
        """
        if self.explainer is None:
            return {"error": "Explainer not initialized"}
        
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])
            
            # Transform using preprocessor
            X_transformed = self.preprocessor.transform(df)
            
            # Generate SHAP values
            shap_values = self.explainer.shap_values(X_transformed)
            
            # Get feature importance for this prediction
            feature_importance = dict(zip(self.feature_names, shap_values[0]))
            
            # Create explanation summary
            explanation = {
                "feature_importance": feature_importance,
                "prediction": float(self.model.predict(X_transformed)[0]),
                "base_value": float(self.explainer.expected_value),
                "feature_names": self.feature_names.tolist()
            }
            
            return explanation
            
        except Exception as e:
            return {"error": f"Error explaining prediction: {e}"}
    
    def create_feature_importance_plot(self, explanation):
        """Create interactive feature importance plot using Plotly."""
        if "error" in explanation:
            return None
        
        # Sort features by absolute SHAP value
        features = explanation["feature_importance"]
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        
        feature_names = [f[0] for f in sorted_features[:10]]  # Top 10 features
        shap_values = [f[1] for f in sorted_features[:10]]
        
        # Create color coding
        colors = ['red' if val < 0 else 'green' for val in shap_values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=shap_values,
                y=feature_names,
                orientation='h',
                marker_color=colors,
                text=[f"${val:,.0f}" for val in shap_values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Feature Impact on Property Value Prediction",
            xaxis_title="SHAP Value (Impact on Price)",
            yaxis_title="Features",
            height=500,
            showlegend=False
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def create_waterfall_plot(self, explanation):
        """Create SHAP waterfall plot."""
        if "error" in explanation:
            return None
        
        features = explanation["feature_importance"]
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        
        feature_names = [f[0] for f in sorted_features[:10]]
        shap_values = [f[1] for f in sorted_features[:10]]
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Property Value",
            orientation="h",
            measure=["relative"] * len(feature_names),
            x=shap_values,
            textposition="outside",
            text=[f"${val:,.0f}" for val in shap_values],
            y=feature_names,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
        ))
        
        fig.update_layout(
            title="SHAP Waterfall Plot - Feature Contributions",
            xaxis_title="SHAP Value",
            yaxis_title="Features",
            height=500,
            showlegend=False
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def get_global_feature_importance(self):
        """Get global feature importance from the model."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                feature_names = self.feature_names
                
                # Sort by importance
                sorted_idx = np.argsort(importance)[::-1]
                
                return {
                    "feature_names": feature_names[sorted_idx].tolist(),
                    "importance": importance[sorted_idx].tolist()
                }
        except Exception as e:
            return {"error": f"Error getting global importance: {e}"}
    
    def create_global_importance_plot(self):
        """Create global feature importance plot."""
        global_importance = self.get_global_feature_importance()
        
        if "error" in global_importance:
            return None
        
        fig = go.Figure(data=[
            go.Bar(
                x=global_importance["importance"][:15],  # Top 15 features
                y=global_importance["feature_names"][:15],
                orientation='h',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title="Global Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=500
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

# Example usage
if __name__ == "__main__":
    explainer = ModelExplainer()
    
    # Example prediction explanation
    sample_property = {
        'bedrooms': 3,
        'bathrooms': 2,
        'parking_spaces': 1,
        'size_sqm': 120,
        'distance_to_city_center': 5.2,
        'distance_to_nearest_school': 0.8,
        'distance_to_nearest_hospital': 2.1,
        'crime_rate': 0.15,
        'area': 'Mount Pleasant',
        'property_type': 'House',
        'condition': 'Good',
        'furnished': 'No',
        'garden': 'Yes',
        'pool': 'No'
    }
    
    explanation = explainer.explain_prediction(sample_property)
    print("Prediction Explanation:", json.dumps(explanation, indent=2)) 