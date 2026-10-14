import os
import joblib
import pandas as pd
import numpy as np

# Global variables for caching loaded model artifacts
_model = None
_preprocessor = None
_feature_names = None

def load_model_artifacts():
    """
    Loads and caches the model, preprocessor, and feature names.
    """
    global _model, _preprocessor, _feature_names
    
    if _model is None or _preprocessor is None or _feature_names is None:
        model_path = "models/model.pkl"
        preprocessor_path = "models/preprocessor.pkl"
        feature_names_path = "models/feature_names.pkl"
        
        if not (os.path.exists(model_path) and os.path.exists(preprocessor_path) and os.path.exists(feature_names_path)):
            raise FileNotFoundError("Model artifacts not found. Please run train_model.py first.")
            
        _model = joblib.load(model_path)
        _preprocessor = joblib.load(preprocessor_path)
        _feature_names = joblib.load(feature_names_path)
        
    return _model, _preprocessor, _feature_names

def predict_churn(customer_data):
    """
    Predicts whether a customer will churn based on customer data.
    
    Parameters:
    customer_data (dict): Dictionary containing the following keys:
        - Age (float/int)
        - Gender (str)
        - Tenure (float/int)
        - Usage Frequency (float/int)
        - Support Calls (float/int)
        - Payment Delay (float/int)
        - Subscription Type (str)
        - Contract Length (str)
        - Total Spend (float/int)
        - Last Interaction (float/int)
        
    Returns:
    dict: A dictionary containing:
        - 'churn_prediction' (int): 0 (Stay) or 1 (Churn)
        - 'churn_probability' (float): Probability of churn (0.0 to 1.0)
        - 'top_features' (list): List of dicts with feature names and global importance
    """
    # Load cached model and preprocessor
    model, preprocessor, feature_names = load_model_artifacts()
    
    # 1. Convert input data to pandas DataFrame with a single row
    # Maintain column ordering of the original raw dataset (excluding CustomerID and Churn)
    raw_feature_order = [
        'Age', 'Gender', 'Tenure', 'Usage Frequency', 
        'Support Calls', 'Payment Delay', 'Subscription Type', 
        'Contract Length', 'Total Spend', 'Last Interaction'
    ]
    
    # Ensure all required features are present in the input
    for f in raw_feature_order:
        if f not in customer_data:
            raise ValueError(f"Missing required feature: {f}")
            
    input_df = pd.DataFrame([customer_data])[raw_feature_order]
    
    # 2. Preprocess features using the preprocessor pipeline
    input_processed = preprocessor.transform(input_df)
    
    # 3. Make prediction
    prediction = int(model.predict(input_processed)[0])
    probabilities = model.predict_proba(input_processed)[0]
    churn_prob = float(probabilities[1])
    
    # 4. Get top 3 features by global feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_features = []
    # Map index to feature name and format the importance value
    for i in range(min(5, len(feature_names))):
        idx = indices[i]
        top_features.append({
            'name': feature_names[idx].replace('cat__', '').replace('num__', ''),
            'importance': float(importances[idx])
        })
        
    return {
        'churn_prediction': prediction,
        'churn_probability': churn_prob,
        'top_features': top_features
    }
