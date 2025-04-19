from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import execute_query
from src.features.engineering import create_time_based_features, create_ratio_features

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn probability",
    version="1.0.0"
)

# Load trained model and preprocessor
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         "models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "models", "preprocessor.pkl")

# Load model and preprocessor
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
except Exception as e:
    model = None
    preprocessor = None
    print(f"Error loading model or preprocessor: {e}")

# Define request data models
class CustomerPredictionRequest(BaseModel):
    customer_id: int

class CustomerBatchPredictionRequest(BaseModel):
    customer_ids: list[int]

class PredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float
    churn_risk_category: str
    risk_factors: list[dict]

# Define API endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to Customer Churn Prediction API"}

@app.get("/health")
def health_check():
    if model is None or preprocessor is None:
        return {"status": "error", "message": "Model or preprocessor not loaded"}
    return {"status": "healthy", "model_version": "1.0.0"}

@app.post("/predict/customer/{customer_id}", response_model=PredictionResponse)
def predict_customer_churn(customer_id: int):
    """
    Predict churn probability for a single customer.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Fetch customer data from database
    query = """
    -- Fetch all customer data using customer ID
    SELECT 
        c.customer_id,
        -- Include all required fields from the feature extraction queries
        -- This would be a simplified version of the full feature extraction query
    FROM 
        customers c
    LEFT JOIN 
        -- Include all necessary joins
    WHERE 
        c.customer_id = :customer_id
    """
    
    try:
        customer_data = execute_query(query, params={"customer_id": customer_id})
        
        if customer_data.empty:
            raise HTTPException(status_code=404, detail=f"Customer with ID {customer_id} not found")
        
        # Apply feature engineering
        customer_features = create_time_based_features(customer_data)
        customer_features = create_ratio_features(customer_features)
        
        # Drop non-feature columns
        X = customer_features.drop(['customer_id', 'churn_status'], axis=1, errors='ignore')
        
        # Preprocess features
        X_processed = preprocessor.transform(X)
        
        # Make prediction
        churn_probability = model.predict_proba(X_processed)[0, 1]
        
        # Determine risk category
        risk_category = "Low Risk"
        if churn_probability >= 0.7:
            risk_category = "High Risk"
        elif churn_probability >= 0.4:
            risk_category = "Medium Risk"
        
        # Get feature importances (if RandomForest or GradientBoosting)
        risk_factors = []
        if hasattr(model, 'feature_importances_'):
            # Get feature names after preprocessing (this is simplified)
            feature_names = X.columns.tolist()
            
            # Combine feature names and importances
            feature_importances = list(zip(feature_names, model.feature_importances_))
            
            # Sort by importance
            feature_importances.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 5 features
            for feature, importance in feature_importances[:5]:
                # Get the value for this customer
                value = X[feature].iloc[0] if feature in X.columns else None
                
                risk_factors.append({
                    "feature": feature,
                    "importance": float(importance),
                    "value": float(value) if value is not None else None
                })
        
        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=float(churn_probability),
            churn_risk_category=risk_category,
            risk_factors=risk_factors
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch_churn(request: CustomerBatchPredictionRequest):
    """
    Predict churn probability for multiple customers.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    for customer_id in request.customer_ids:
        try:
            prediction = predict_customer_churn(customer_id)
            results.append(prediction)
        except HTTPException as e:
            if e.status_code == 404:
                # Skip customers that don't exist
                continue
            else:
                raise e
    
    return results
