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
customer_support AS (
    SELECT 
        customer_id,
        COUNT(*) AS ticket_count,
        SUM(CASE WHEN priority = 'high' THEN 1 ELSE 0 END) AS high_priority_tickets,
        AVG(resolution_time_hours) AS avg_resolution_time
    FROM 
        support_tickets
    GROUP BY 
        customer_id
),
usage_patterns AS (
    SELECT 
        customer_id,
        AVG(daily_usage_minutes) AS avg_daily_usage,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY daily_usage_minutes) AS median_usage,
        STDDEV(daily_usage_minutes) AS usage_variability,
        COUNT(DISTINCT DATE_TRUNC('month', usage_date)) AS active_months
    FROM 
        product_usage
    WHERE 
        usage_date >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY 
        customer_id
),
subscription_details AS (
    SELECT 
        s.customer_id,
        s.plan_id,
        s.monthly_fee,
        s.start_date,
        s.end_date,
        EXTRACT(DAY FROM (COALESCE(s.end_date, CURRENT_DATE) - s.start_date)) AS subscription_tenure_days,
        COUNT(h.change_date) AS plan_changes,
        CASE WHEN s.end_date IS NULL THEN 0 ELSE 1 END AS churned
    FROM 
        subscriptions s
    LEFT JOIN 
        subscription_history h ON s.customer_id = h.customer_id
    GROUP BY 
        s.customer_id, s.plan_id, s.monthly_fee, s.start_date, s.end_date
)
SELECT 
    c.customer_id,
    c.age,
    c.gender,
    c.city,
    c.state,
    c.country,
    c.join_date,
    EXTRACT(DAY FROM (CURRENT_DATE - c.join_date)) AS customer_tenure_days,
    t.transaction_count,
    t.total_spent,
    t.avg_transaction_value,
    t.unique_products_purchased,
    t.last_transaction_date,
    EXTRACT(DAY FROM (CURRENT_DATE - t.last_transaction_date)) AS days_since_last_transaction,
    s.ticket_count,
    s.high_priority_tickets,
    s.avg_resolution_time,
    u.avg_daily_usage,
    u.median_usage,
    u.usage_variability,
    u.active_months,
    sd.plan_id,
    sd.monthly_fee,
    sd.subscription_tenure_days,
    sd.plan_changes,
    sd.churned AS churn_status
FROM 
    customers c
LEFT JOIN 
    customer_transactions t ON c.customer_id = t.customer_id
LEFT JOIN 
    customer_support s ON c.customer_id = s.customer_id
LEFT JOIN 
    usage_patterns u ON c.customer_id = u.customer_id
LEFT JOIN 
    subscription_details sd ON c.customer_id = sd.customer_id
WHERE 
    c.join_date <= CURRENT_DATE - INTERVAL '3 months'
ORDER BY 
    c.customer_id;
