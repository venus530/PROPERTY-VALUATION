import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def load_data(filepath):
    """Loads data from a CSV file."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None

def preprocess_data(df):
    """Preprocesses the data."""
    # Define categorical and numerical features
    categorical_features = ['Property Type', 'Location', 'Swimming Pool', 'Boundary', 'Age Category']
    numerical_features = ['Area', 'Number of structures', 'Land Rate', 'Number of rooms', 
                            'Proximity to schools(km)', 'Proximity to healthcare(km)', 'Proximity to malls(km)', 
                            'Proximity to highway(km)']

    # Define the target variable
    target = 'Market Value'

    # Separate features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Create preprocessing pipelines for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # Apply the preprocessing pipeline to the data
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Trains and evaluates multiple regression models with hyperparameter tuning."""
    
    # Define hyperparameter grids for each model
    param_grids = {
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {}  # Linear Regression has no hyperparameters to tune
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'Support Vector Machine': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
        }
    }

    results = {}
    trained_models = {}

    for name, config in param_grids.items():
        print(f"--- {name} ---")
        
        if name == 'Linear Regression':
            # For Linear Regression, just fit the model directly
            model = config['model']
        model.fit(X_train, y_train)
            best_model = model
            print("Linear Regression fitted with default parameters")
        else:
            # For other models, perform hyperparameter tuning
            print("Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {-grid_search.best_score_:.2f}")
        
        # Make predictions with the best model
        y_pred = best_model.predict(X_test)
        
        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R-squared (R²): {r2:.2f}")
        print("-" * 20)

        results[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}
        trained_models[name] = best_model
        
    return results, trained_models

def visualize_model_performance(results):
    """Creates bar plots to compare model performance metrics."""
    results_df = pd.DataFrame(results).T
    
    # Plotting R-squared
    plt.figure(figsize=(10, 6))
    sns.barplot(x=results_df.index, y=results_df['R2'])
    plt.title('Model Comparison: R-squared (R²)')
    plt.xlabel('Model')
    plt.ylabel('R-squared (R²)')
    plt.savefig('model_comparison_r2.png')
    plt.show()

    # Plotting MAE
    plt.figure(figsize=(10, 6))
    sns.barplot(x=results_df.index, y=results_df['MAE'])
    plt.title('Model Comparison: Mean Absolute Error (MAE)')
    plt.xlabel('Model')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.savefig('model_comparison_mae.png')
    plt.show()

def visualize_feature_importance(model, model_name, preprocessor):
    """Visualizes feature importances for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        print(f"Feature importance not available for {model_name}")
        return

    importances = model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(20) # Top 20 features

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title(f'Feature Importance: {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.replace(" ", "_")}.png')
    plt.show()

def plot_predictions_vs_actuals(model, model_name, X_test, y_test):
    """Creates a scatter plot of predicted vs actual values."""
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.title(f'Predicted vs. Actual Values ({model_name})')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.savefig(f'predicted_vs_actual_{model_name.replace(" ", "_")}.png')
    plt.show()

def save_best_model(trained_models, results, preprocessor, model_name=None):
    """Saves the best performing model and preprocessor."""
    
    # If no specific model is requested, find the best one based on R² score
    if model_name is None:
        best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
        print(f"\nBest model based on R² score: {best_model_name}")
    else:
        best_model_name = model_name
        if best_model_name not in trained_models:
            print(f"Error: Model '{best_model_name}' not found in trained models")
            return None
    
    # Get the best model
    best_model = trained_models[best_model_name]
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the best model
    model_filename = f'models/best_model_{best_model_name.replace(" ", "_").lower()}.joblib'
    joblib.dump(best_model, model_filename)
    print(f"Best model saved as: {model_filename}")
    
    # Save the preprocessor
    preprocessor_filename = 'models/preprocessor.joblib'
    joblib.dump(preprocessor, preprocessor_filename)
    print(f"Preprocessor saved as: {preprocessor_filename}")
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'performance': results[best_model_name],
        'feature_names': preprocessor.get_feature_names_out().tolist(),
        'categorical_features': ['Property Type', 'Location', 'Swimming Pool', 'Boundary', 'Age Category'],
        'numerical_features': ['Area', 'Number of structures', 'Land Rate', 'Number of rooms', 
                              'Proximity to schools(km)', 'Proximity to healthcare(km)', 'Proximity to malls(km)', 
                              'Proximity to highway(km)'],
        'target': 'Market Value'
    }
    
    metadata_filename = 'models/model_metadata.json'
    import json
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata saved as: {metadata_filename}")
    
    return best_model_name, model_filename, preprocessor_filename

def load_saved_model(model_path, preprocessor_path):
    """Loads a saved model and preprocessor."""
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def main():
    """Main function to run the property valuation framework."""
    # 1. Load data
    data_filepath = 'PROPERTYVALUATIONS.csv'
    df = load_data(data_filepath)

    if df is not None:
        # 2. Preprocess data
        X, y, preprocessor = preprocess_data(df)

        # 3. Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Train and evaluate models
        results, trained_models = train_and_evaluate_models(X_train, y_train, X_test, y_test)

        # 5. Save the best model
        best_model_info = save_best_model(trained_models, results, preprocessor)
        
        if best_model_info:
            best_model_name, model_path, preprocessor_path = best_model_info
            print(f"\n✅ Best model '{best_model_name}' has been saved successfully!")
            print(f"📁 Model files saved in 'models/' directory:")
            print(f"   - Model: {model_path}")
            print(f"   - Preprocessor: {preprocessor_path}")
            print(f"   - Metadata: models/model_metadata.json")
            print(f"\n🚀 You can now use this model in your application!")

        # 6. Visualize results
        visualize_model_performance(results)

        for name, model in trained_models.items():
            plot_predictions_vs_actuals(model, name, X_test, y_test)
            if name in ['Random Forest', 'Gradient Boosting']:
                visualize_feature_importance(model, name, preprocessor)

if __name__ == '__main__':
    main()
