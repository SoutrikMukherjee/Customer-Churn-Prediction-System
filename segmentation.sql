WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.join_date)) AS tenure_years,
        COALESCE(SUM(t.amount), 0) AS total_revenue,
        COALESCE(COUNT(t.transaction_id), 0) AS transaction_count,
        COALESCE(SUM(t.amount) / NULLIF(EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.join_date)), 0), 0) AS annual_revenue,
        COALESCE(MAX(t.transaction_date), c.join_date) AS last_activity_date,
        EXTRACT(DAY FROM (CURRENT_DATE - COALESCE(MAX(t.transaction_date), c.join_date))) AS days_since_last_activity,
        COALESCE(AVG(u.daily_usage_minutes), 0) AS avg_usage,
        COALESCE(COUNT(DISTINCT s.ticket_id), 0) AS support_ticket_count
    FROM 
        customers c
    LEFT JOIN 
        transactions t ON c.customer_id = t.customer_id
    LEFT JOIN 
        product_usage u ON c.customer_id = u.customer_id
    LEFT JOIN 
        support_tickets s ON c.customer_id = s.customer_id
    GROUP BY 
        c.customer_id, c.join_date
)
SELECT 
    customer_id,
    CASE
        WHEN days_since_last_activity <= 30 AND transaction_count > 10 AND annual_revenue > 500 THEN 'High Value Active'
        WHEN days_since_last_activity <= 30 AND transaction_count > 0 THEN 'Active'
        WHEN days_since_last_activity BETWEEN 31 AND 90 THEN 'At Risk'
        WHEN days_since_last_activity BETWEEN 91 AND 180 THEN 'Needs Attention'
        WHEN days_since_last_activity > 180 THEN 'Inactive'
        ELSE 'New' 
    END AS customer_segment,
    tenure_years,
    total_revenue,
    transaction_count,
    annual_revenue,
    days_since_last_activity,
    avg_usage,
    support_ticket_count,
    NTILE(4) OVER (ORDER BY annual_revenue DESC) AS revenue_quartile,
    NTILE(4) OVER (ORDER BY avg_usage DESC) AS usage_quartile,
    NTILE(4) OVER (ORDER BY days_since_last_activity) AS recency_quartile
FROM 
    customer_metrics
ORDER BY 
    annual_revenue DESC;
