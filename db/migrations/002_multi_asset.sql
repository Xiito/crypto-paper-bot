-- ============================================================
-- Migration 002: Multi-Asset Support (Stocks/ETFs + Crypto)
-- ============================================================
-- Adds asset_class and broker columns to track which market
-- and broker each trade originated from.
-- ============================================================

-- Add asset_class column to trades
ALTER TABLE trades ADD COLUMN IF NOT EXISTS asset_class VARCHAR(10)
    DEFAULT 'crypto' CHECK (asset_class IN ('crypto', 'stock', 'etf'));

-- Add broker column to trades
ALTER TABLE trades ADD COLUMN IF NOT EXISTS broker VARCHAR(20)
    DEFAULT 'binance';

-- Add asset_class to sessions for mixed-market sessions
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS asset_classes VARCHAR(50)
    DEFAULT 'crypto';

-- Index for filtering trades by asset class
CREATE INDEX IF NOT EXISTS idx_trades_asset_class ON trades(asset_class);

-- Index for filtering trades by broker
CREATE INDEX IF NOT EXISTS idx_trades_broker ON trades(broker);
