import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import pickle
import os

def train_test_validation_split(X, y, test_size=0.2, val_size=0.25, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    test_size : float
        Proportion of data to include in the test split
    val_size : float
        Proportion of the remaining data to include in the validation split
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: training + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: training vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, 
        random_state=random_state, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(model, X, y, model_name=None):
    """
    Evaluate a trained model on a dataset.
    
    Parameters:
    -----------
    model : trained model object
        The trained model to evaluate
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    model_name : str, optional
        Name of the model
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
    }
    
    # Add ROC AUC if probability predictions are available
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y, y_prob)
    
    # Add model name if provided
    if model_name:
        metrics['model'] = model_name
    
    # Print summary
    print(f"Model: {model_name if model_name else 'Unknown'}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    return metrics

def train_random_forest(X_train, y_train, X_val, y_val, param_grid=None, cv=5):
    """
    Train a Random Forest classifier with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Training target variable
    X_val : pandas.DataFrame or numpy.ndarray
        Validation features
    y_val : pandas.Series or numpy.ndarray
        Validation target variable
    param_grid : dict, optional
        Grid of hyperparameters to search
    cv : int, optional
        Number of cross-validation folds
        
    Returns:
    --------
    tuple
        (best_model, validation_metrics)
    """
    # Default hyperparameter grid
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 0.9, 1.0]
        }
    
    # Initialize base model
    gb = GradientBoostingClassifier(random_state=42)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv),
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Train with grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on validation set
    val_metrics = evaluate_model(best_model, X_val, y_val, "Gradient Boosting")
    
    # Print best parameters
    print(f"Best Parameters: {grid_search.best_params_}")
    
    return best_model, val_metrics

def train_logistic_regression(X_train, y_train, X_val, y_val, param_grid=None, cv=5):
    """
    Train a Logistic Regression classifier with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Training target variable
    X_val : pandas.DataFrame or numpy.ndarray
        Validation features
    y_val : pandas.Series or numpy.ndarray
        Validation target variable
    param_grid : dict, optional
        Grid of hyperparameters to search
    cv : int, optional
        Number of cross-validation folds
        
    Returns:
    --------
    tuple
        (best_model, validation_metrics)
    """
    # Default hyperparameter grid
    if param_grid is None:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'class_weight': ['balanced', None]
        }
    
    # Initialize base model
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv),
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Train with grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on validation set
    val_metrics = evaluate_model(best_model, X_val, y_val, "Logistic Regression")
    
    # Print best parameters
    print(f"Best Parameters: {grid_search.best_params_}")
    
    return best_model, val_metrics

def save_model(model, filename):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : trained model object
        The trained model to save
    filename : str
        Path to save the model
    """
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filename}")

def load_model(filename):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    filename : str
        Path to the saved model
        
    Returns:
    --------
    trained model object
        The loaded model
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {filename}")
    return model

def train_and_evaluate_models(X, y, preprocessor, test_size=0.2, val_size=0.25, random_state=42):
    """
    Train and evaluate multiple models on the provided dataset.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Target variable
    preprocessor : sklearn.compose.ColumnTransformer
        Preprocessing pipeline
    test_size : float
        Proportion of data to include in the test split
    val_size : float
        Proportion of the remaining data to include in the validation split
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with trained models and their evaluation metrics
    """
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_split(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    # Apply preprocessing
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train models
    print("Training Random Forest...")
    rf_model, rf_val_metrics = train_random_forest(X_train_processed, y_train, X_val_processed, y_val)
    
    print("\nTraining Gradient Boosting...")
    gb_model, gb_val_metrics = train_gradient_boosting(X_train_processed, y_train, X_val_processed, y_val)
    
    print("\nTraining Logistic Regression...")
    lr_model, lr_val_metrics = train_logistic_regression(X_train_processed, y_train, X_val_processed, y_val)
    
    # Evaluate models on test set
    print("\n--- Test Set Evaluation ---")
    rf_test_metrics = evaluate_model(rf_model, X_test_processed, y_test, "Random Forest")
    gb_test_metrics = evaluate_model(gb_model, X_test_processed, y_test, "Gradient Boosting")
    lr_test_metrics = evaluate_model(lr_model, X_test_processed, y_test, "Logistic Regression")
    
    # Return results
    return {
        'models': {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'logistic_regression': lr_model
        },
        'preprocessor': preprocessor,
        'validation_metrics': {
            'random_forest': rf_val_metrics,
            'gradient_boosting': gb_val_metrics,
            'logistic_regression': lr_val_metrics
        },
        'test_metrics': {
            'random_forest': rf_test_metrics,
            'gradient_boosting': gb_test_metrics,
            'logistic_regression': lr_test_metrics
        }
    }
