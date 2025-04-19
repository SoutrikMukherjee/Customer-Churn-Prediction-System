# Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.9-blue)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.2-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0.0-green)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0.0-red)

## Project Overview

This project demonstrates an end-to-end machine learning workflow for predicting customer churn, with a focus on SQL for data extraction and preprocessing. The system connects to a relational database, leverages SQL queries for data manipulation, and implements machine learning models to predict which customers are likely to churn.

## Key Features

- **SQL-based Data Pipeline**: Direct integration with PostgreSQL database
- **Advanced SQL Operations**: Complex joins, aggregations, and window functions
- **Exploratory Data Analysis**: Statistical analysis and visualization of customer behavior
- **Feature Engineering**: Transform raw data into ML-ready features
- **ML Model Implementation**: Train and evaluate multiple models with cross-validation
- **Model Deployment**: Simple API for making predictions on new data

## Project Structure

```
customer-churn-prediction/
├── data/
│   ├── schema.sql                 # Database schema definition
│   └── sample_data.sql            # Sample data insertion scripts
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb  # EDA notebook
│   └── 02_model_development.ipynb     # Model training notebook
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration parameters
│   ├── data/
│   │   ├── __init__.py
│   │   ├── database.py            # Database connection utilities
│   │   ├── extraction.py          # SQL query functions
│   │   └── preprocessing.py       # Data cleaning and transformation
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py         # Feature creation functions
│   │   └── selection.py           # Feature selection algorithms
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py               # Model training functions
│   │   ├── evaluate.py            # Model evaluation utilities
│   │   └── predict.py             # Prediction functions
│   └── visualization/
│       ├── __init__.py
│       └── plots.py               # Visualization functions
├── api/
│   ├── __init__.py
│   ├── app.py                     # FastAPI application
│   └── endpoints.py               # API endpoints
├── tests/
│   ├── __init__.py
│   ├── test_data.py               # Tests for data processing
│   ├── test_features.py           # Tests for feature engineering
│   └── test_models.py             # Tests for model performance
├── requirements.txt               # Project dependencies
├── setup.py                       # Package installation
└── README.md                      # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.9+
- PostgreSQL 13+
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the database:
```bash
psql -U postgres -f data/schema.sql
psql -U postgres -f data/sample_data.sql
```

5. Update the database configuration in `src/config.py` with your credentials.

## Usage

### Data Extraction and Analysis

Run the Jupyter notebooks to explore the data and develop models:

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### Running the API

Start the prediction API:

```bash
uvicorn api.app:app --reload
```

The API will be available at http://localhost:8000.

## Implementation Details

### Database Schema

The project uses a relational database with the following main tables:

- `customers`: Basic customer information
- `subscriptions`: Service subscription details
- `transactions`: Customer transaction history
- `support_tickets`: Customer support interactions
- `product_usage`: Product usage statistics
