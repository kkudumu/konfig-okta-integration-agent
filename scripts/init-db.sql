-- Database initialization script for Konfig
-- This script sets up the PostgreSQL database with the pgvector extension

-- Enable pgvector extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable uuid-ossp extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pg_trgm extension for text similarity search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create database user if not exists (for development)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'konfig') THEN
        CREATE ROLE konfig WITH LOGIN PASSWORD 'konfig_dev_password';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE konfig TO konfig;
GRANT ALL ON SCHEMA public TO konfig;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO konfig;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO konfig;

-- Set default permissions for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO konfig;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO konfig;

-- Create a function to update the updated_at column automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create a function to generate random UUIDs (if not using uuid-ossp)
CREATE OR REPLACE FUNCTION gen_random_uuid_fallback()
RETURNS UUID AS $$
BEGIN
    RETURN uuid_generate_v4();
EXCEPTION
    WHEN undefined_function THEN
        -- Fallback if uuid-ossp is not available
        RETURN (SELECT encode(gen_random_bytes(16), 'hex')::uuid);
END;
$$ language 'plpgsql';

-- Verify extensions are installed
SELECT 
    'vector' as extension, 
    (SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector') > 0 as installed
UNION ALL
SELECT 
    'uuid-ossp' as extension, 
    (SELECT COUNT(*) FROM pg_extension WHERE extname = 'uuid-ossp') > 0 as installed
UNION ALL
SELECT 
    'pg_trgm' as extension, 
    (SELECT COUNT(*) FROM pg_extension WHERE extname = 'pg_trgm') > 0 as installed;