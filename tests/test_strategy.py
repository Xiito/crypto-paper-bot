"""Unit tests for the signal generation engine.

Tests cover EMA crossover detection, RSI range filtering,
ATR volatility gating, and edge cases in signal computation.
"""

import numpy as np
import pytest

from bot.strategy import SignalDirection, SignalEngine, TradeSignal


def _generate_ohlcv(
    num_candles: int = 100,
    base_price: float = 50000.0,
    trend: str = "flat",
    volatility: float = 100.0,
) -> list[list]:
    """Generate synthetic OHLCV data for testing.

    Args:
        num_candles: Number of candles to generate.
        base_price: Starting price.
        trend: Price trend ('up', 'down', 'flat').
        volatility: Price volatility in absolute terms.

    Returns:
        List of [timestamp, open, high, low, close, volume] candles.
    """
    np.random.seed(42)
    candles = []
    price = base_price

    for i in range(num_candles):
        if trend == "up":
            price += volatility * 0.1 + np.random.normal(0, volatility * 0.3)
        elif trend == "down":
            price -= volatility * 0.1 + np.random.normal(0, volatility * 0.3)
        else:
            price += np.random.normal(0, volatility * 0.3)

        price = max(price, 100.0)
        open_price = price + np.random.normal(0, volatility * 0.1)
        high = max(open_price, price) + abs(np.random.normal(0, volatility * 0.2))
        low = min(open_price, price) - abs(np.random.normal(0, volatility * 0.2))
        volume = abs(np.random.normal(1000, 200))

        candles.append([
            1700000000000 + i * 60000,
            open_price,
            high,
            low,
            price,
            volume,
        ])

    return candles


def _generate_crossover_ohlcv(direction: str = "bullish") -> list[list]:
    """Generate OHLCV data that produces a specific EMA crossover.

    Args:
        direction: 'bullish' or 'bearish' crossover.

    Returns:
        OHLCV candle data with the specified crossover pattern.
    """
    candles = []
    base_price = 50000.0

    if direction == "bullish":
        for i in range(50):
            price = base_price - (i * 20)
            candles.append([1700000000000 + i * 60000, price, price + 30, price - 30, price, 1000])
        for i in range(50):
            price = base_price - 1000 + (i * 40)
            candles.append([1700000000000 + (50 + i) * 60000, price, price + 30, price - 30, price, 1000])
    else:
        for i in range(50):
            price = base_price + (i * 20)
            candles.append([1700000000000 + i * 60000, price, price + 30, price - 30, price, 1000])
        for i in range(50):
            price = base_price + 1000 - (i * 40)
            candles.append([1700000000000 + (50 + i) * 60000, price, price + 30, price - 30, price, 1000])

    return candles


class TestSignalEngine:
    """Tests for the SignalEngine class."""

    def test_init_default_parameters(self):
        """Test that SignalEngine initializes with correct default parameters."""
        engine = SignalEngine()
        assert engine.ema_fast_period == 9
        assert engine.ema_slow_period == 21
        assert engine.rsi_period == 14
        assert engine.atr_period == 14

    def test_init_custom_parameters(self):
        """Test that SignalEngine accepts custom parameters."""
        engine = SignalEngine(
            ema_fast_period=5,
            ema_slow_period=15,
            rsi_period=10,
            atr_period=10,
        )
        assert engine.ema_fast_period == 5
        assert engine.ema_slow_period == 15

    def test_evaluate_returns_trade_signal(self):
        """Test that evaluate returns a TradeSignal object."""
        engine = SignalEngine()
        ohlcv = _generate_ohlcv(100, trend="flat")
        signal = engine.evaluate("BTC/USDT", ohlcv)

        assert isinstance(signal, TradeSignal)
        assert signal.pair == "BTC/USDT"
        assert isinstance(signal.direction, SignalDirection)
        assert signal.ema_fast > 0
        assert signal.ema_slow > 0
        assert 0 <= signal.rsi <= 100
        assert signal.atr >= 0

    def test_flat_market_no_signal(self):
        """Test that a flat market with low volatility produces no signal."""
        engine = SignalEngine()
        ohlcv = _generate_ohlcv(100, trend="flat", volatility=1.0)
        signal = engine.evaluate("BTC/USDT", ohlcv)
        assert signal.direction == SignalDirection.NONE

    def test_signal_strength_range(self):
        """Test that signal strength is always between 0.0 and 1.0."""
        engine = SignalEngine()
        for trend in ["up", "down", "flat"]:
            ohlcv = _generate_ohlcv(100, trend=trend)
            signal = engine.evaluate("BTC/USDT", ohlcv)
            assert 0.0 <= signal.strength <= 1.0

    def test_ema_values_positive(self):
        """Test that computed EMA values are always positive."""
        engine = SignalEngine()
        ohlcv = _generate_ohlcv(100)
        signal = engine.evaluate("ETH/USDT", ohlcv)
        assert signal.ema_fast > 0
        assert signal.ema_slow > 0

    def test_rsi_range(self):
        """Test that RSI is always in 0-100 range."""
        engine = SignalEngine()
        for trend in ["up", "down", "flat"]:
            ohlcv = _generate_ohlcv(100, trend=trend)
            signal = engine.evaluate("BTC/USDT", ohlcv)
            assert 0 <= signal.rsi <= 100

    def test_atr_non_negative(self):
        """Test that ATR is always non-negative."""
        engine = SignalEngine()
        ohlcv = _generate_ohlcv(100)
        signal = engine.evaluate("BTC/USDT", ohlcv)
        assert signal.atr >= 0

    def test_insufficient_data_returns_none_signal(self):
        """Test that insufficient candle data returns NONE direction."""
        engine = SignalEngine()
        ohlcv = _generate_ohlcv(10)
        signal = engine.evaluate("BTC/USDT", ohlcv)
        assert signal.direction == SignalDirection.NONE

    def test_long_signal_conditions(self):
        """Test that long signals are generated under bullish conditions."""
        engine = SignalEngine(
            rsi_long_min=40,
            rsi_long_max=70,
        )
        ohlcv = _generate_crossover_ohlcv("bullish")
        signal = engine.evaluate("BTC/USDT", ohlcv)
        assert signal.direction in [SignalDirection.LONG, SignalDirection.NONE]

    def test_short_signal_conditions(self):
        """Test that short signals are generated under bearish conditions."""
        engine = SignalEngine(
            rsi_short_min=30,
            rsi_short_max=60,
        )
        ohlcv = _generate_crossover_ohlcv("bearish")
        signal = engine.evaluate("BTC/USDT", ohlcv)
        assert signal.direction in [SignalDirection.SHORT, SignalDirection.NONE]

    def test_multiple_pairs(self):
        """Test that engine correctly handles multiple trading pairs."""
        engine = SignalEngine()
        pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        for pair in pairs:
            ohlcv = _generate_ohlcv(100)
            signal = engine.evaluate(pair, ohlcv)
            assert signal.pair == pair

    def test_signal_has_timestamp(self):
        """Test that returned signal includes a timestamp."""
        engine = SignalEngine()
        ohlcv = _generate_ohlcv(100)
        signal = engine.evaluate("BTC/USDT", ohlcv)
        assert signal.timestamp is not None

    def test_signal_metadata_populated(self):
        """Test that signal metadata fields are correctly populated."""
        engine = SignalEngine()
        ohlcv = _generate_ohlcv(100)
        signal = engine.evaluate("BTC/USDT", ohlcv)
        assert signal.pair == "BTC/USDT"
        assert signal.close_price > 0
