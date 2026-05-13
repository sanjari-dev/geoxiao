CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS strategy_dna (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ast JSONB NOT NULL,
    params JSONB NOT NULL,
    fitness_score DOUBLE PRECISION,
    status VARCHAR DEFAULT 'candidate',
    oos_profit_factor DOUBLE PRECISION,
    regime_profile JSONB,
    generation_born INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trial_logs (
    id BIGSERIAL PRIMARY KEY,
    strategy_id UUID NOT NULL REFERENCES strategy_dna(id),
    window_id INTEGER NOT NULL,
    regime_id INTEGER NOT NULL,
    profit_factor DOUBLE PRECISION,
    total_pips DOUBLE PRECISION,
    trade_count INTEGER,
    is_oos BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS trade_logs (
    id BIGSERIAL PRIMARY KEY,
    strategy_id UUID NOT NULL REFERENCES strategy_dna(id),
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ NOT NULL,
    direction VARCHAR NOT NULL,
    gross_pips DOUBLE PRECISION,
    net_pips DOUBLE PRECISION,
    window_type VARCHAR NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_strategy_dna_fitness
    ON strategy_dna(fitness_score DESC);

CREATE INDEX IF NOT EXISTS idx_trial_logs_strategy
    ON trial_logs(strategy_id);
