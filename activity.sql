SELECT 
    customer_id,
    -- Recent usage trend using window functions
    AVG(daily_usage_minutes) OVER(
        PARTITION BY customer_id 
        ORDER BY usage_date DESC 
        ROWS BETWEEN 0 AND 6
    ) AS recent_7day_avg_usage,
    
    AVG(daily_usage_minutes) OVER(
        PARTITION BY customer_id 
        ORDER BY usage_date DESC 
        ROWS BETWEEN 7 AND 13
    ) AS prev_7day_avg_usage,
    
    -- Calculate relative change in usage
    (
        AVG(daily_usage_minutes) OVER(
            PARTITION BY customer_id 
            ORDER BY usage_date DESC 
            ROWS BETWEEN 0 AND 6
        ) - 
        AVG(daily_usage_minutes) OVER(
            PARTITION BY customer_id 
            ORDER BY usage_date DESC 
            ROWS BETWEEN 7 AND 13
        )
    ) / NULLIF(
        AVG(daily_usage_minutes) OVER(
            PARTITION BY customer_id 
            ORDER BY usage_date DESC 
            ROWS BETWEEN 7 AND 13
        ), 0
    ) * 100 AS usage_percent_change,
    
    -- Detect usage pattern changes
    STDDEV(daily_usage_minutes) OVER(
        PARTITION BY customer_id 
        ORDER BY usage_date DESC 
        ROWS BETWEEN 0 AND 29
    ) AS recent_usage_variability,
    
    -- Feature for weekend vs weekday usage patterns
    AVG(CASE WHEN EXTRACT(DOW FROM usage_date) IN (0, 6) 
             THEN daily_usage_minutes ELSE 0 END) OVER(
        PARTITION BY customer_id
        ORDER BY usage_date DESC 
        ROWS BETWEEN 0 AND 29
    ) / NULLIF(
        AVG(CASE WHEN EXTRACT(DOW FROM usage_date) NOT IN (0, 6) 
                 THEN daily_usage_minutes ELSE 0 END) OVER(
            PARTITION BY customer_id
            ORDER BY usage_date DESC 
            ROWS BETWEEN 0 AND 29
        ), 0
    ) AS weekend_weekday_ratio
FROM 
    product_usage
WHERE 
    usage_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY 
    customer_id, usage_date, daily_usage_minutes
ORDER BY 
    customer_id, usage_date DESC;
