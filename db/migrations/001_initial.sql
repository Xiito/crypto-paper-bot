-- ============================================================
-- Migration 001: Initial schema
-- Applied at: first deployment
-- ============================================================
-- This migration creates all tables required by the crypto
-- paper trading bot using IF NOT EXISTS guards so it is safe
-- to re-run without errors.
-- ============================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- -----------------------------------------------------------
-- Sessions table
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date                DATE NOT NULL UNIQUE,
    starting_capital    DECIMAL(12,2) NOT NULL DEFAULT 1000.00,
    ending_capital      DECIMAL(12,2),
    total_trades        INTEGER NOT NULL DEFAULT 0,
    win_count           INTEGER NOT NULL DEFAULT 0,
    loss_count          INTEGER NOT NULL DEFAULT 0,
    session_return_pct  DECIMAL(8,4),
    regime_label        VARCHAR(30),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_date   ON sessions(date DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_regime ON sessions(regime_label);

-- -----------------------------------------------------------
-- Trades table
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS trades (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id            UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    pair                  VARCHAR(20) NOT NULL,
    direction             VARCHAR(5)  NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
    entry_price           DECIMAL(18,8) NOT NULL,
    exit_price            DECIMAL(18,8),
    quantity              DECIMAL(18,8) NOT NULL,
    signal_ema_cross      BOOLEAN NOT NULL,
    signal_rsi            DECIMAL(6,2) NOT NULL,
    signal_atr            DECIMAL(18,8) NOT NULL,
    regime_tag            VARCHAR(30),
    pnl                   DECIMAL(12,4),
    pnl_pct               DECIMAL(8,4),
    hold_duration_seconds INTEGER,
    exit_reason           VARCHAR(30) CHECK (exit_reason IN (
        'STOP_LOSS', 'TAKE_PROFIT', 'SESSION_CLOSE', 'SIGNAL_REVERSAL', 'HARD_STOP'
    )),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id);
CREATE INDEX IF NOT EXISTS idx_trades_pair    ON trades(pair);
CREATE INDEX IF NOT EXISTS idx_trades_regime  ON trades(regime_tag);
CREATE INDEX IF NOT EXISTS idx_trades_created ON trades(created_at DESC);

-- -----------------------------------------------------------
-- Reflections table
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS reflections (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id           UUID NOT NULL UNIQUE REFERENCES sessions(id) ON DELETE CASCADE,
    regime_label         VARCHAR(30) NOT NULL,
    regime_confidence    DECIMAL(5,2) NOT NULL,
    losses_analysis      JSONB,
    wins_analysis        JSONB,
    parameter_suggestion JSONB,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reflections_session ON reflections(session_id);
CREATE INDEX IF NOT EXISTS idx_reflections_regime  ON reflections(regime_label);

-- -----------------------------------------------------------
-- Regime performance table
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS regime_performance (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    regime_label   VARCHAR(30) NOT NULL UNIQUE,
    total_sessions INTEGER      NOT NULL DEFAULT 0,
    avg_return_pct DECIMAL(8,4) NOT NULL DEFAULT 0,
    win_rate       DECIMAL(5,2) NOT NULL DEFAULT 0,
    last_updated   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_regime_perf_label ON regime_performance(regime_label);

-- -----------------------------------------------------------
-- Parameter history table
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS parameter_history (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parameter_name     VARCHAR(50)  NOT NULL,
    old_value          VARCHAR(50)  NOT NULL,
    new_value          VARCHAR(50)  NOT NULL,
    suggested_by_agent BOOLEAN      NOT NULL DEFAULT TRUE,
    applied_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_param_hist_name    ON parameter_history(parameter_name);
CREATE INDEX IF NOT EXISTS idx_param_hist_applied ON parameter_history(applied_at DESC);
