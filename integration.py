import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from src.config import DATABASE_URI

def get_database_connection():
    """Establish connection to the PostgreSQL database."""
    try:
        engine = create_engine(DATABASE_URI)
        connection = engine.connect()
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

def execute_query(query, params=None):
    """Execute SQL query and return results as pandas DataFrame."""
    connection = get_database_connection()
    try:
        if params:
            result = pd.read_sql_query(sa.text(query), connection, params=params)
        else:
            result = pd.read_sql_query(sa.text(query), connection)
        return result
    except Exception as e:
        print(f"Error executing query: {e}")
        raise
    finally:
        connection.close()

def get_customer_features():
    """Extract customer features for churn prediction."""
    query = """
    -- Complex customer feature query (as shown above)
    WITH customer_transactions AS (
        SELECT 
            customer_id,
            COUNT(*) AS transaction_count,
            SUM(amount) AS total_spent,
            AVG(amount) AS avg_transaction_value,
            MAX(transaction_date) AS last_transaction_date,
            MIN(transaction_date) AS first_transaction_date,
            COUNT(DISTINCT product_id) AS unique_products_purchased
        FROM 
            transactions
        GROUP BY 
            customer_id
    ),
    -- Additional CTEs as shown in the above SQL queries
    -- ...
    
    -- Final SELECT statement that joins all the CTEs
    SELECT 
        c.customer_id,
        c.age,
        c.gender,
        c.city,
        c.state,
        -- Other fields as shown in the query above
        sd.churned AS churn_status
    FROM 
        customers c
    LEFT JOIN 
        customer_transactions t ON c.customer_id = t.customer_id
    -- Other joins as shown in the query above
    WHERE 
        c.join_date <= CURRENT_DATE - INTERVAL '3 months'
    ORDER BY 
        c.customer_id;
    """
    return execute_query(query)

def get_recent_customer_activity(days=30):
    """Get recent customer activity data."""
    query = """
    -- Recent customer activity query as shown above
    SELECT 
        customer_id,
        -- Recent usage trend using window functions
        AVG(daily_usage_minutes) OVER(
            PARTITION BY customer_id 
            ORDER BY usage_date DESC 
            ROWS BETWEEN 0 AND 6
        ) AS recent_7day_avg_usage,
        -- Other calculations as shown in the query above
    FROM 
        product_usage
    WHERE 
        usage_date >= CURRENT_DATE - INTERVAL :days days
    GROUP BY 
        customer_id, usage_date, daily_usage_minutes
    ORDER BY 
        customer_id, usage_date DESC;
    """
    return execute_query(query, params={"days": days})

def get_customer_segments():
    """Get customer segmentation data."""
    query = """
    -- Customer segmentation query as shown above
    WITH customer_metrics AS (
        SELECT 
            c.customer_id,
            EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.join_date)) AS tenure_years,
            -- Other calculations as shown in the query above
        FROM 
            customers c
        LEFT JOIN 
            transactions t ON c.customer_id = t.customer_id
        -- Other joins as shown in the query above
        GROUP BY 
            c.customer_id, c.join_date
    )
    SELECT 
        customer_id,
        CASE
            WHEN days_since_last_activity <= 30 AND transaction_count > 10 AND annual_revenue > 500 THEN 'High Value Active'
            -- Other case statements as shown in the query above
        END AS customer_segment,
        -- Other fields as shown in the query above
    FROM 
        customer_metrics
    ORDER BY 
        annual_revenue DESC;
    """
    return execute_query(query)
