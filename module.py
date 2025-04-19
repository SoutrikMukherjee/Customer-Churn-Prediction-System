import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_time_based_features(df):
    """
    Create time-based features from date columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with date columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional time-based features
    """
    df = df.copy()
    
    # Customer tenure features
    if 'join_date' in df.columns:
        df['join_date'] = pd.to_datetime(df['join_date'])
        df['customer_tenure_days'] = (pd.Timestamp.now() - df['join_date']).dt.days
        df['customer_tenure_months'] = df['customer_tenure_days'] / 30
        df['customer_tenure_years'] = df['customer_tenure_days'] / 365
        
        # Extract temporal components
        df['join_month'] = df['join_date'].dt.month
        df['join_quarter'] = df['join_date'].dt.quarter
        df['join_year'] = df['join_date'].dt.year
        df['join_day_of_week'] = df['join_date'].dt.dayofweek
        df['join_is_weekend'] = df['join_day_of_week'].isin([5, 6]).astype(int)
    
    # Transaction recency features
    if 'last_transaction_date' in df.columns:
        df['last_transaction_date'] = pd.to_datetime(df['last_transaction_date'])
        df['days_since_last_transaction'] = (pd.Timestamp.now() - df['last_transaction_date']).dt.days
        df['months_since_last_transaction'] = df['days_since_last_transaction'] / 30
        
        # Transaction frequency if we have first_transaction_date
        if 'first_transaction_date' in df.columns:
            df['first_transaction_date'] = pd.to_datetime(df['first_transaction_date'])
            active_days = (df['last_transaction_date'] - df['first_transaction_date']).dt.days
            active_days = np.maximum(active_days, 1)  # Avoid division by zero
            df['transaction_frequency'] = df['transaction_count'] / active_days
    
    # Subscription features
    if 'subscription_tenure_days' in df.columns:
        df['subscription_tenure_months'] = df['subscription_tenure_days'] / 30
        df['subscription_tenure_years'] = df['subscription_tenure_days'] / 365
    
    return df

def create_ratio_features(df):
    """
    Create ratio features from existing numerical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with numerical columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional ratio features
    """
    df = df.copy()
    
    # Financial ratios
    if all(col in df.columns for col in ['total_spent', 'transaction_count']):
        df['avg_transaction_amount'] = df['total_spent'] / np.maximum(df['transaction_count'], 1)
    
    if all(col in df.columns for col in ['total_spent', 'customer_tenure_days']):
        df['daily_spend'] = df['total_spent'] / np.maximum(df['customer_tenure_days'], 1)
        df['monthly_spend'] = df['daily_spend'] * 30
        df['annual_spend'] = df['daily_spend'] * 365
    
    # Product engagement ratios
    if all(col in df.columns for col in ['avg_daily_usage', 'median_usage']):
        df['usage_skew_ratio'] = df['avg_daily_usage'] / np.maximum(df['median_usage'], 1)
    
    # Support interaction ratios
    if all(col in df.columns for col in ['ticket_count', 'customer_tenure_months']):
        df['tickets_per_month'] = df['ticket_count'] / np.maximum(df['customer_tenure_months'], 1)
    
    if all(col in df.columns for col in ['high_priority_tickets', 'ticket_count']):
        df['high_priority_ticket_ratio'] = df['high_priority_tickets'] / np.maximum(df['ticket_count'], 1)
    
    return df

def build_preprocessing_pipeline(df):
    """
    Build a preprocessing pipeline for feature transformation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to determine column types
        
    Returns:
    --------
    sklearn.compose.ColumnTransformer
        Preprocessing pipeline
    """
    # Identify column types
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target variable and ID fields from features
    for col in ['customer_id', 'churn_status']:
        if col in numeric_features:
            numeric_features.remove(col)
        if col in categorical_features:
            categorical_features.remove(col)
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def prepare_features_for_modeling(df):
    """
    Prepare features for machine learning modeling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw features DataFrame
        
    Returns:
    --------
    tuple
        (X, y, feature_names, preprocessor)
    """
    # Apply feature engineering
    df = create_time_based_features(df)
    df = create_ratio_features(df)
    
    # Separate features and target
    X = df.drop(['customer_id', 'churn_status'], axis=1, errors='ignore')
    y = df['churn_status'] if 'churn_status' in df.columns else None
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(X)
    
    # Get original feature names (before transformation)
    feature_names = X.columns.tolist()
    
    return X, y, feature_names, preprocessor
