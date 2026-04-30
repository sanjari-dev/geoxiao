-- File: src/data/migrations/001_initial_schema.sql
-- Jalankan via: alembic upgrade head

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =========================================================
-- TABEL 1: strategy_dna  (Blueprint §2.1)
-- Representasi genetik setiap individu GP
-- =========================================================
CREATE TABLE strategy_dna (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    generation      INTEGER NOT NULL,
    individual_id   VARCHAR(64) NOT NULL,
    tree_repr       TEXT NOT NULL,
    tree_depth      SMALLINT NOT NULL,
    tree_nodes      SMALLINT NOT NULL,
    params_json     JSONB NOT NULL DEFAULT '{}',
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','backtesting',
                                      'passed','eliminated','archived')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (generation, individual_id)
);
CREATE INDEX idx_dna_generation ON strategy_dna(generation);
CREATE INDEX idx_dna_status     ON strategy_dna(status);
CREATE INDEX idx_dna_symbol_tf  ON strategy_dna(symbol, timeframe);

-- =========================================================
-- TABEL 2: trial_logs  (Blueprint §2.2)
-- Hasil setiap trial Optuna per strategi
-- =========================================================
CREATE TABLE trial_logs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id         UUID NOT NULL
                        REFERENCES strategy_dna(id) ON DELETE CASCADE,
    optuna_trial_id     INTEGER NOT NULL,
    study_name          VARCHAR(128) NOT NULL,
    params_json         JSONB NOT NULL,
    profit_factor       NUMERIC(10,4),
    total_pips          NUMERIC(10,2),
    max_drawdown_pips   NUMERIC(10,2),
    trade_count         INTEGER,
    fitness_score       NUMERIC(10,6),
    eliminated_reason   VARCHAR(128),
    duration_sec        NUMERIC(8,2),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_trial_strategy ON trial_logs(strategy_id);
CREATE INDEX idx_trial_fitness  ON trial_logs(fitness_score DESC);
CREATE INDEX idx_trial_pf       ON trial_logs(profit_factor DESC);

-- =========================================================
-- TABEL 3: trade_logs  (Blueprint §2.3)
-- Log setiap trade — net_pips adalah GENERATED COLUMN
-- =========================================================
CREATE TABLE trade_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trial_id        UUID NOT NULL
                    REFERENCES trial_logs(id) ON DELETE CASCADE,
    strategy_id     UUID NOT NULL
                    REFERENCES strategy_dna(id) ON DELETE CASCADE,
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(4) NOT NULL CHECK (side IN ('BUY','SELL')),
    order_type      VARCHAR(16) NOT NULL
                    CHECK (order_type IN ('LIMIT','STOP_LIMIT')),
    entry_price     NUMERIC(18,5) NOT NULL,
    exit_price      NUMERIC(18,5),
    sl_price        NUMERIC(18,5) NOT NULL,
    tp_price        NUMERIC(18,5),
    entry_time      TIMESTAMPTZ NOT NULL,
    exit_time       TIMESTAMPTZ,
    raw_pips        NUMERIC(8,2),
    spread_pips     NUMERIC(6,2) NOT NULL DEFAULT 0,
    slippage_pips   NUMERIC(6,2) NOT NULL DEFAULT 0,
    commission_pips NUMERIC(6,2) NOT NULL DEFAULT 0,
    net_pips        NUMERIC(8,2)
                    GENERATED ALWAYS AS
                    (raw_pips - spread_pips - slippage_pips - commission_pips)
                    STORED,
    exit_reason     VARCHAR(32),
    backtest_month  DATE NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_trade_trial   ON trade_logs(trial_id);
CREATE INDEX idx_trade_strategy ON trade_logs(strategy_id);
CREATE INDEX idx_trade_month   ON trade_logs(backtest_month);
CREATE INDEX idx_trade_symbol  ON trade_logs(symbol, backtest_month);

-- =========================================================
-- TABEL 4: monthly_metrics  (Blueprint §2.4)
-- Agregasi per-bulan — output dari MetricsCalculator
-- =========================================================
CREATE TABLE monthly_metrics (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trial_id            UUID NOT NULL
                        REFERENCES trial_logs(id) ON DELETE CASCADE,
    strategy_id         UUID NOT NULL
                        REFERENCES strategy_dna(id) ON DELETE CASCADE,
    backtest_month      DATE NOT NULL,
    trade_count         INTEGER NOT NULL,
    winning_trades      INTEGER NOT NULL,
    losing_trades       INTEGER NOT NULL,
    gross_profit        NUMERIC(10,2) NOT NULL,
    gross_loss          NUMERIC(10,2) NOT NULL,
    net_pips            NUMERIC(10,2) NOT NULL,
    profit_factor       NUMERIC(10,4),
    max_drawdown_pips   NUMERIC(10,2) NOT NULL,
    win_rate            NUMERIC(5,4),
    avg_risk_pips       NUMERIC(8,2),
    constraint_passed   BOOLEAN NOT NULL DEFAULT FALSE,
    elimination_flags   JSONB,
    calculated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (trial_id, backtest_month)
);
CREATE INDEX idx_monthly_trial   ON monthly_metrics(trial_id);
CREATE INDEX idx_monthly_passed  ON monthly_metrics(constraint_passed);
