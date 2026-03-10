-- Pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Passages table
CREATE TABLE IF NOT EXISTS passages (
    id BIGINT PRIMARY KEY,
    text TEXT NOT NULL,
    title TEXT
);

-- Queries table
CREATE TABLE IF NOT EXISTS queries (
    id BIGINT PRIMARY KEY,
    text TEXT NOT NULL
);

-- Qrels table
CREATE TABLE IF NOT EXISTS qrels (
    id SERIAL PRIMARY KEY,
    query_id BIGINT REFERENCES queries(id),
    passage_id BIGINT REFERENCES passages(id),
    relevance INT NOT NULL
);

-- SPLADE table
CREATE TABLE IF NOT EXISTS splade (
    passage_id BIGINT PRIMARY KEY REFERENCES passages(id),
    term_weights JSONB NOT NULL
);

-- COLBERT table
CREATE TABLE IF NOT EXISTS colbert (
    passage_id BIGINT PRIMARY KEY REFERENCES passages(id),
    embedding vector[] NOT NULL
);

-- DPR table
CREATE TABLE IF NOT EXISTS dpr (
    passage_id BIGINT PRIMARY KEY REFERENCES passages(id),
    embedding vector(768) NOT NULL
);

-- Search logs table
CREATE TABLE IF NOT EXISTS search_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    algorithm VARCHAR(50),
    query TEXT NOT NULL,
    latency_ms FLOAT NOT NULL
);

-- Results table
CREATE TABLE IF NOT EXISTS results (
    id SERIAL PRIMARY KEY,
    search_log_id BIGINT REFERENCES search_logs(id),
    algorithm VARCHAR(50) NOT NULL,
    passage_id BIGINT REFERENCES passages(id),
    rank INT NOT NULL,
    score FLOAT NOT NULL
);

-- Indexes for performance
-- Foreign key indexes (improve JOIN performance)
CREATE INDEX IF NOT EXISTS idx_qrels_query_id ON qrels(query_id);
CREATE INDEX IF NOT EXISTS idx_qrels_passage_id ON qrels(passage_id);
CREATE INDEX IF NOT EXISTS idx_results_passage_id ON results(passage_id);

-- Full-text search index on passages
CREATE INDEX IF NOT EXISTS idx_passages_text ON passages USING gin (to_tsvector('english', text));

-- SPLADE sparse vector search
CREATE INDEX IF NOT EXISTS idx_splade_term_weights ON splade USING gin (term_weights);

-- DPR dense vector search (approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_dpr_embedding ON dpr USING ivfflat (embedding vector_cosine_ops);

-- Search logs indexes for analytics
CREATE INDEX IF NOT EXISTS idx_search_logs_timestamp ON search_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_search_logs_algorithm ON search_logs(algorithm);