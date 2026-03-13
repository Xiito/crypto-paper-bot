"""Unit tests for the risk management module.

Tests cover position sizing, max drawdown enforcement,
daily loss limits, and emergency stop logic.
"""

import pytest

from bot.risk_manager import RiskManager


class TestRiskManager:
    """Tests for the RiskManager class."""

    def test_init_default_parameters(self):
        """Test RiskManager initializes with correct defaults."""
        rm = RiskManager(capital=10000.0)
        assert rm.capital == 10000.0
        assert rm.max_risk_per_trade_pct == 0.02
        assert rm.max_daily_loss_pct == 0.05
        assert rm.max_drawdown_pct == 0.15

    def test_init_custom_parameters(self):
        """Test RiskManager accepts custom parameters."""
        rm = RiskManager(
            capital=50000.0,
            max_risk_per_trade_pct=0.01,
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10,
        )
        assert rm.capital == 50000.0
        assert rm.max_risk_per_trade_pct == 0.01

    def test_position_size_basic(self):
        """Test basic position size calculation."""
        rm = RiskManager(capital=10000.0, max_risk_per_trade_pct=0.02)
        size = rm.calculate_position_size(
            entry_price=50000.0,
            stop_loss_price=49000.0,
            atr=500.0,
        )
        assert size > 0
        assert isinstance(size, float)

    def test_position_size_zero_atr(self):
        """Test position size with zero ATR returns zero or minimum."""
        rm = RiskManager(capital=10000.0)
        size = rm.calculate_position_size(
            entry_price=50000.0,
            stop_loss_price=49000.0,
            atr=0.0,
        )
        assert size >= 0

    def test_position_size_respects_max_risk(self):
        """Test that position size never exceeds max risk per trade."""
        rm = RiskManager(capital=10000.0, max_risk_per_trade_pct=0.02)
        size = rm.calculate_position_size(
            entry_price=50000.0,
            stop_loss_price=48000.0,
            atr=1000.0,
        )
        max_risk_usd = 10000.0 * 0.02
        risk_per_unit = abs(50000.0 - 48000.0)
        max_size_from_risk = max_risk_usd / risk_per_unit
        assert size <= max_size_from_risk * 1.001

    def test_can_trade_initially_true(self):
        """Test that trading is allowed initially."""
        rm = RiskManager(capital=10000.0)
        assert rm.can_trade() is True

    def test_daily_loss_limit_triggers_stop(self):
        """Test that exceeding daily loss limit blocks trading."""
        rm = RiskManager(capital=10000.0, max_daily_loss_pct=0.05)
        rm.record_trade_pnl(-600.0)
        assert rm.can_trade() is False

    def test_daily_loss_within_limit_allows_trade(self):
        """Test that trading remains allowed within daily loss limit."""
        rm = RiskManager(capital=10000.0, max_daily_loss_pct=0.05)
        rm.record_trade_pnl(-200.0)
        assert rm.can_trade() is True

    def test_drawdown_limit_triggers_stop(self):
        """Test that exceeding drawdown limit blocks trading."""
        rm = RiskManager(capital=10000.0, max_drawdown_pct=0.15)
        rm.update_capital(8000.0)
        assert rm.can_trade() is False

    def test_drawdown_within_limit_allows_trade(self):
        """Test that trading remains allowed within drawdown limit."""
        rm = RiskManager(capital=10000.0, max_drawdown_pct=0.15)
        rm.update_capital(9000.0)
        assert rm.can_trade() is True

    def test_reset_daily_stats(self):
        """Test that daily stats reset correctly."""
        rm = RiskManager(capital=10000.0, max_daily_loss_pct=0.05)
        rm.record_trade_pnl(-600.0)
        assert rm.can_trade() is False
        rm.reset_daily_stats()
        assert rm.can_trade() is True

    def test_record_trade_pnl_accumulates(self):
        """Test that PnL records accumulate correctly."""
        rm = RiskManager(capital=10000.0, max_daily_loss_pct=0.05)
        rm.record_trade_pnl(-100.0)
        rm.record_trade_pnl(-200.0)
        rm.record_trade_pnl(-150.0)
        assert rm.daily_pnl == pytest.approx(-450.0)

    def test_positive_pnl_does_not_block_trading(self):
        """Test that positive PnL does not affect trading ability."""
        rm = RiskManager(capital=10000.0)
        rm.record_trade_pnl(500.0)
        rm.record_trade_pnl(300.0)
        assert rm.can_trade() is True

    def test_update_capital_increases_drawdown_baseline(self):
        """Test that updating capital to a new high updates the baseline."""
        rm = RiskManager(capital=10000.0, max_drawdown_pct=0.15)
        rm.update_capital(12000.0)
        rm.update_capital(10500.0)
        assert rm.can_trade() is True

    def test_position_size_scales_with_capital(self):
        """Test that position size scales proportionally with capital."""
        rm_small = RiskManager(capital=10000.0, max_risk_per_trade_pct=0.02)
        rm_large = RiskManager(capital=100000.0, max_risk_per_trade_pct=0.02)

        size_small = rm_small.calculate_position_size(
            entry_price=50000.0, stop_loss_price=49000.0, atr=500.0
        )
        size_large = rm_large.calculate_position_size(
            entry_price=50000.0, stop_loss_price=49000.0, atr=500.0
        )
        assert size_large > size_small

    def test_emergency_stop_blocks_all_trading(self):
        """Test that emergency stop flag blocks all trading."""
        rm = RiskManager(capital=10000.0)
        rm.trigger_emergency_stop("Test emergency")
        assert rm.can_trade() is False

    def test_max_position_size_cap(self):
        """Test that position size is capped at max position size limit."""
        rm = RiskManager(
            capital=10000.0,
            max_risk_per_trade_pct=0.02,
            max_position_size_pct=0.10,
        )
        size = rm.calculate_position_size(
            entry_price=1.0,
            stop_loss_price=0.99,
            atr=0.01,
        )
        max_position_usd = 10000.0 * 0.10
        assert size * 1.0 <= max_position_usd * 1.001
