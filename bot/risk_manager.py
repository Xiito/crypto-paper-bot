"""Risk management module for position sizing, loss caps, and trade validation.

Enforces the following rules:
- Maximum 2% capital risk per trade
- Maximum 3 concurrent open positions
- Daily hard stop at -10% session loss
- Dynamic position sizing based on ATR stop-loss distance
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from bot.strategy import SignalDirection, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of a position sizing calculation."""

    approved: bool
    quantity: float
    risk_amount: float
    stop_loss_distance: float
    rejection_reason: Optional[str] = None


@dataclass
class OpenPosition:
    """Tracks an open position for risk accounting."""

    trade_id: str
    pair: str
    direction: str
    entry_price: float
    quantity: float
    stop_loss_price: float
    unrealized_pnl: float = 0.0


class RiskManager:
    """Manages trade risk: position sizing, exposure limits, and daily loss cap.

    Validates every trade request against capital constraints, position limits,
    and the daily drawdown hard stop before allowing execution.
    """

    def __init__(
        self,
        starting_capital: float = 1000.0,
        max_risk_per_trade_pct: float = 0.02,
        max_concurrent_positions: int = 3,
        daily_hard_stop_pct: float = -0.10,
    ) -> None:
        """Initialize the risk manager.

        Args:
            starting_capital: Session starting capital in USDT.
            max_risk_per_trade_pct: Maximum capital percentage risked per trade (0.02 = 2%).
            max_concurrent_positions: Maximum number of simultaneously open positions.
            daily_hard_stop_pct: Daily loss threshold to halt trading (-0.10 = -10%).
        """
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.daily_hard_stop_pct = daily_hard_stop_pct
        self._open_positions: dict[str, OpenPosition] = {}
        self._realized_pnl: float = 0.0
        self._total_trades: int = 0
        self._hard_stop_triggered: bool = False

    @property
    def open_position_count(self) -> int:
        """Return the number of currently open positions."""
        return len(self._open_positions)

    @property
    def session_return_pct(self) -> float:
        """Return the current session return as a decimal fraction."""
        if self.starting_capital == 0:
            return 0.0
        return (self.current_capital - self.starting_capital) / self.starting_capital

    @property
    def is_hard_stop_triggered(self) -> bool:
        """Return whether the daily hard stop has been triggered."""
        return self._hard_stop_triggered

    def validate_trade(self, signal: TradeSignal) -> PositionSizeResult:
        """Validate a trade signal against all risk constraints and calculate position size.

        Checks in order:
        1. Daily hard stop not triggered
        2. Signal has a valid direction
        3. Open positions under limit
        4. Not already in a position for this pair
        5. Sufficient capital for the trade

        Args:
            signal: The trade signal to validate.

        Returns:
            PositionSizeResult with approval status and calculated position size.
        """
        if self._hard_stop_triggered:
            return PositionSizeResult(
                approved=False,
                quantity=0.0,
                risk_amount=0.0,
                stop_loss_distance=0.0,
                rejection_reason="Daily hard stop triggered (session loss exceeded threshold)",
            )

        if signal.direction == SignalDirection.NONE:
            return PositionSizeResult(
                approved=False,
                quantity=0.0,
                risk_amount=0.0,
                stop_loss_distance=0.0,
                rejection_reason="No valid signal direction",
            )

        if self.open_position_count >= self.max_concurrent_positions:
            return PositionSizeResult(
                approved=False,
                quantity=0.0,
                risk_amount=0.0,
                stop_loss_distance=0.0,
                rejection_reason=f"Max concurrent positions reached ({self.max_concurrent_positions})",
            )

        for pos in self._open_positions.values():
            if pos.pair == signal.pair:
                return PositionSizeResult(
                    approved=False,
                    quantity=0.0,
                    risk_amount=0.0,
                    stop_loss_distance=0.0,
                    rejection_reason=f"Already have an open position for {signal.pair}",
                )

        stop_loss_distance = abs(signal.current_price - signal.stop_loss_price)
        if stop_loss_distance <= 0:
            return PositionSizeResult(
                approved=False,
                quantity=0.0,
                risk_amount=0.0,
                stop_loss_distance=0.0,
                rejection_reason="Invalid stop-loss distance (zero or negative)",
            )

        risk_amount = self.current_capital * self.max_risk_per_trade_pct
        quantity = risk_amount / stop_loss_distance

        position_value = quantity * signal.current_price
        if position_value > self.current_capital * 0.95:
            quantity = (self.current_capital * 0.95) / signal.current_price
            risk_amount = quantity * stop_loss_distance

        if quantity <= 0 or risk_amount <= 0:
            return PositionSizeResult(
                approved=False,
                quantity=0.0,
                risk_amount=0.0,
                stop_loss_distance=stop_loss_distance,
                rejection_reason="Insufficient capital for minimum position size",
            )

        logger.info(
            "Trade approved: %s %s, qty=%.8f, risk=$%.2f (%.1f%% of capital), SL_dist=%.8f",
            signal.direction.value,
            signal.pair,
            quantity,
            risk_amount,
            (risk_amount / self.current_capital) * 100,
            stop_loss_distance,
        )

        return PositionSizeResult(
            approved=True,
            quantity=quantity,
            risk_amount=risk_amount,
            stop_loss_distance=stop_loss_distance,
        )

    def register_open_position(
        self,
        trade_id: str,
        pair: str,
        direction: str,
        entry_price: float,
        quantity: float,
        stop_loss_price: float,
    ) -> None:
        """Register a newly opened position for risk tracking.

        Args:
            trade_id: Unique trade identifier.
            pair: Trading pair.
            direction: 'LONG' or 'SHORT'.
            entry_price: Entry price.
            quantity: Position quantity.
            stop_loss_price: Stop-loss price.
        """
        self._open_positions[trade_id] = OpenPosition(
            trade_id=trade_id,
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss_price=stop_loss_price,
        )
        self._total_trades += 1
        logger.info("Registered position %s: %s %s @ %.8f, qty=%.8f", trade_id, direction, pair, entry_price, quantity)

    def close_position(self, trade_id: str, exit_price: float) -> float:
        """Close an open position and update capital.

        Args:
            trade_id: The trade identifier to close.
            exit_price: The exit price.

        Returns:
            The realized P&L for this trade.

        Raises:
            KeyError: If the trade_id is not found in open positions.
        """
        if trade_id not in self._open_positions:
            raise KeyError(f"Trade {trade_id} not found in open positions")

        pos = self._open_positions.pop(trade_id)

        if pos.direction == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        self._realized_pnl += pnl
        self.current_capital += pnl

        logger.info(
            "Closed position %s: %s %s, entry=%.8f, exit=%.8f, pnl=%.4f, capital=%.2f",
            trade_id, pos.direction, pos.pair, pos.entry_price, exit_price, pnl, self.current_capital,
        )

        self._check_hard_stop()
        return pnl

    def update_unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        """Update unrealized P&L for all open positions.

        Args:
            current_prices: Dict mapping pair symbols to current prices.

        Returns:
            Total unrealized P&L across all open positions.
        """
        total_unrealized = 0.0
        for pos in self._open_positions.values():
            price = current_prices.get(pos.pair)
            if price is None:
                continue
            if pos.direction == "LONG":
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - price) * pos.quantity
            total_unrealized += pos.unrealized_pnl
        return total_unrealized

    def check_stop_losses(self, current_prices: dict[str, float]) -> list[str]:
        """Check all open positions against their stop-loss prices.

        Args:
            current_prices: Dict mapping pair symbols to current prices.

        Returns:
            List of trade IDs that have hit their stop-loss.
        """
        triggered = []
        for trade_id, pos in list(self._open_positions.items()):
            price = current_prices.get(pos.pair)
            if price is None:
                continue
            if pos.direction == "LONG" and price <= pos.stop_loss_price:
                triggered.append(trade_id)
                logger.warning("Stop-loss triggered for LONG %s: price=%.8f <= SL=%.8f", pos.pair, price, pos.stop_loss_price)
            elif pos.direction == "SHORT" and price >= pos.stop_loss_price:
                triggered.append(trade_id)
                logger.warning("Stop-loss triggered for SHORT %s: price=%.8f >= SL=%.8f", pos.pair, price, pos.stop_loss_price)
        return triggered

    def get_open_positions(self) -> list[OpenPosition]:
        """Return all currently open positions.

        Returns:
            List of OpenPosition objects.
        """
        return list(self._open_positions.values())

    def get_session_stats(self) -> dict:
        """Return current session statistics.

        Returns:
            Dict with session return, realized P&L, open positions, etc.
        """
        return {
            "starting_capital": self.starting_capital,
            "current_capital": self.current_capital,
            "realized_pnl": self._realized_pnl,
            "session_return_pct": self.session_return_pct,
            "total_trades": self._total_trades,
            "open_positions": self.open_position_count,
            "hard_stop_triggered": self._hard_stop_triggered,
        }

    def reset(self, starting_capital: float = 1000.0) -> None:
        """Reset the risk manager for a new session.

        Args:
            starting_capital: New session starting capital.
        """
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self._open_positions.clear()
        self._realized_pnl = 0.0
        self._total_trades = 0
        self._hard_stop_triggered = False
        logger.info("Risk manager reset with capital $%.2f", starting_capital)

    def _check_hard_stop(self) -> None:
        """Check if the daily hard stop threshold has been breached."""
        if self.session_return_pct <= self.daily_hard_stop_pct:
            self._hard_stop_triggered = True
            logger.critical(
                "DAILY HARD STOP TRIGGERED: session return %.2f%% <= threshold %.2f%%",
                self.session_return_pct * 100,
                self.daily_hard_stop_pct * 100,
            )
