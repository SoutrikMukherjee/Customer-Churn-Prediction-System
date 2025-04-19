DROP TABLE IF EXISTS product_usage;
DROP TABLE IF EXISTS support_tickets;
DROP TABLE IF EXISTS transaction_items;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS subscription_history;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS plans;
DROP TABLE IF EXISTS products;

-- Create plans table
CREATE TABLE plans (
    plan_id SERIAL PRIMARY KEY,
    plan_name VARCHAR(100) NOT NULL,
    plan_description TEXT,
    monthly_fee DECIMAL(10, 2) NOT NULL,
    billing_cycle INTEGER NOT NULL DEFAULT 1, -- In months
    has_trial BOOLEAN DEFAULT FALSE,
    trial_days INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create products table
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    product_description TEXT,
    product_category VARCHAR(100),
    price DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create customers table
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(50),
    age INTEGER,
    gender VARCHAR(20),
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100),
    postal_code VARCHAR(20),
    join_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    referral_source VARCHAR(100),
    marketing_consent BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create subscriptions table
CREATE TABLE subscriptions (
    subscription_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    plan_id INTEGER NOT NULL REFERENCES plans(plan_id),
    start_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    end_date TIMESTAMP,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    monthly_fee DECIMAL(10, 2) NOT NULL,
    billing_cycle INTEGER NOT NULL DEFAULT 1, -- In months
    next_billing_date TIMESTAMP,
    payment_method VARCHAR(50),
    auto_renew BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create subscription history table
CREATE TABLE subscription_history (
    history_id SERIAL PRIMARY KEY,
    subscription_id INTEGER NOT NULL REFERENCES subscriptions(subscription_id),
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    old_plan_id INTEGER REFERENCES plans(plan_id),
    new_plan_id INTEGER REFERENCES plans(plan_id),
    change_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    change_type VARCHAR(50) NOT NULL, -- 'new', 'upgrade', 'downgrade', 'cancel', 'renew'
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create transactions table
CREATE TABLE transactions (
    transaction_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    subscription_id INTEGER REFERENCES subscriptions(subscription_id),
    transaction_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    amount DECIMAL(10, 2) NOT NULL,
    payment_method VARCHAR(50),
    payment_status VARCHAR(50) NOT NULL,
    transaction_type VARCHAR(50) NOT NULL, -- 'subscription', 'one-time', 'refund'
    currency VARCHAR(10) DEFAULT 'USD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    product_id INTEGER REFERENCES products(product_id) -- For one-time purchases
);

-- Create transaction items table (for itemized purchases)
CREATE TABLE transaction_items (
    item_id SERIAL PRIMARY KEY,
    transaction_id INTEGER NOT NULL REFERENCES transactions(transaction_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price DECIMAL(10, 2) NOT NULL,
    discount DECIMAL(10, 2) DEFAULT 0,
    total_price DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create support tickets table
CREATE TABLE support_tickets (
    ticket_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    subject VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    priority VARCHAR(50) DEFAULT 'medium',
    status VARCHAR(50) DEFAULT 'open',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution_time_hours DECIMAL(10, 2),
    agent_id INTEGER,
    satisfaction_rating INTEGER -- 1-5 scale
);

-- Create product usage table
CREATE TABLE product_usage (
    usage_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    subscription_id INTEGER REFERENCES subscriptions(subscription_id),
    usage_date DATE NOT NULL,
    daily_usage_minutes DECIMAL(10, 2) NOT NULL,
    feature_usage JSONB, -- Stores usage of specific features
    device_type VARCHAR(100),
    os_type VARCHAR(100),
    browser_type VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_customers_join_date ON customers(join_date);
CREATE INDEX idx_subscriptions_customer_id ON subscriptions(customer_id);
CREATE INDEX idx_subscriptions_status ON subscriptions(status);
CREATE INDEX idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_support_tickets_customer_id ON support_tickets(customer_id);
CREATE INDEX idx_product_usage_customer_date ON product_usage(customer_id, usage_date);

-- Add comments
COMMENT ON TABLE customers IS 'Stores customer account information';
COMMENT ON TABLE subscriptions IS 'Tracks customer subscription plans';
COMMENT ON TABLE subscription_history IS 'Records changes to customer subscriptions';
COMMENT ON TABLE transactions IS 'Records all financial transactions';
COMMENT ON TABLE support_tickets IS 'Tracks customer support interactions';
COMMENT ON TABLE product_usage IS 'Records customer product usage patterns';
