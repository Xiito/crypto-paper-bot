"""Trading session lifecycle manager.

Orchestrates the full trading loop: signal generation, regime filtering,
risk validation, order execution, and position monitoring.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from bot.order_executor import OrderExecutor, OrderResult
from bot.regime_classifier import RegimeClassifier, MarketRegime
from bot.risk_manager import RiskManager
from bot.strategy import SignalEngine, SignalDirection
from bot.trade_logger import TradeLogger

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Configuration for a trading session."""

    pairs: list[str]
    timeframe: str = "1m"
    candle_limit: int = 100
    poll_interval_seconds: float = 60.0
    starting_capital: float = 1000.0
    max_risk_per_trade_pct: float = 0.02
    max_concurrent_positions: int = 3
    daily_hard_stop_pct: float = -0.10
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    bb_period: int = 20
    adx_threshold: float = 25.0
    bb_width_threshold: float = 0.02


class SessionManager:
    """Manages the full lifecycle of a paper trading session.

    Coordinates all bot components — signal engine, regime classifier,
    risk manager, order executor, and trade logger — into a continuous
    polling loop.
    """

    def __init__(
        self,
        config: SessionConfig,
        executor: OrderExecutor,
        trade_logger: TradeLogger,
    ) -> None:
        """Initialize the session manager.

        Args:
            config: Session configuration parameters.
            executor: Initialized OrderExecutor instance.
            trade_logger: Initialized TradeLogger instance.
        """
        self.config = config
        self.executor = executor
        self.trade_logger = trade_logger

        self.signal_engine = SignalEngine(
            ema_fast_period=config.ema_fast_period,
            ema_slow_period=config.ema_slow_period,
            rsi_period=config.rsi_period,
            atr_period=config.atr_period,
        )
        self.regime_classifier = RegimeClassifier(
            adx_period=config.adx_period,
            bb_period=config.bb_period,
            adx_threshold=config.adx_threshold,
            bb_width_threshold=config.bb_width_threshold,
        )
        self.risk_manager = RiskManager(
            starting_capital=config.starting_capital,
            max_risk_per_trade_pct=config.max_risk_per_trade_pct,
            max_concurrent_positions=config.max_concurrent_positions,
            daily_hard_stop_pct=config.daily_hard_stop_pct,
        )
        self._running = False
        self._open_trades: dict[str, dict] = {}

    async def run(self) -> None:
        """Start the trading session loop.

        Polls all configured pairs at the configured interval until
        the daily hard stop is triggered or the session is stopped.
        """
        self._running = True
        logger.info("Session started: pairs=%s, timeframe=%s", self.config.pairs, self.config.timeframe)

        try:
            while self._running:
                if self.risk_manager.is_hard_stop_triggered:
                    logger.critical("Hard stop active — halting session")
                    break

                current_prices = await self._fetch_current_prices()
                self._monitor_stop_losses(current_prices)
                self.risk_manager.update_unrealized_pnl(current_prices)

                for pair in self.config.pairs:
                    if not self._running:
                        break
                    await self._process_pair(pair)

                stats = self.risk_manager.get_session_stats()
                logger.info(
                    "Cycle complete | capital=%.2f | return=%.2f%% | open=%d | trades=%d",
                    stats["current_capital"],
                    stats["session_return_pct"] * 100,
                    stats["open_positions"],
                    stats["total_trades"],
                )

                await asyncio.sleep(self.config.poll_interval_seconds)

        except asyncio.CancelledError:
            logger.info("Session cancelled")
        finally:
            self._running = False
            logger.info("Session ended")

    def stop(self) -> None:
        """Signal the session loop to stop after the current cycle."""
        self._running = False
        logger.info("Stop signal sent to session manager")

    async def _process_pair(self, pair: str) -> None:
        """Process a single trading pair: fetch data, classify, signal, execute.

        Args:
            pair: Trading pair symbol to process.
        """
        try:
            ohlcv = await self.executor.fetch_ohlcv(pair, self.config.timeframe, self.config.candle_limit)

            regime = self.regime_classifier.classify(ohlcv)
            if not regime.is_tradeable:
                logger.debug("Skipping %s — regime: %s (ADX=%.1f)", pair, regime.regime.value, regime.adx)
                return

            signal = self.signal_engine.evaluate(pair, ohlcv)
            if signal.direction == SignalDirection.NONE:
                return

            size_result = self.risk_manager.validate_trade(signal)
            if not size_result.approved:
                logger.debug("Trade rejected for %s: %s", pair, size_result.rejection_reason)
                return

            order = await self.executor.execute_market_order(pair, signal.direction.value, size_result.quantity)
            if not order.success:
                logger.error("Order failed for %s: %s", pair, order.error)
                return

            trade_id = order.order_id or f"TRADE_{pair}_{int(time.time())}"
            self.risk_manager.register_open_position(
                trade_id=trade_id,
                pair=pair,
                direction=signal.direction.value,
                entry_price=order.price,
                quantity=order.quantity,
                stop_loss_price=signal.stop_loss_price,
            )
            self._open_trades[trade_id] = {
                "pair": pair,
                "direction": signal.direction.value,
                "entry_price": order.price,
                "quantity": order.quantity,
                "stop_loss_price": signal.stop_loss_price,
            }
            self.trade_logger.log_entry(trade_id, signal, size_result, order)

        except Exception as exc:
            logger.exception("Error processing pair %s: %s", pair, exc)

    async def _fetch_current_prices(self) -> dict[str, float]:
        """Fetch current prices for all pairs with open positions.

        Returns:
            Dict mapping pair symbols to current prices.
        """
        prices = {}
        open_pairs = {v["pair"] for v in self._open_trades.values()}
        for pair in open_pairs:
            try:
                ticker = await self.executor.fetch_ticker(pair)
                prices[pair] = float(ticker["last"])
            except Exception as exc:
                logger.warning("Failed to fetch price for %s: %s", pair, exc)
        return prices

    def _monitor_stop_losses(self, current_prices: dict[str, float]) -> None:
        """Check and execute stop-losses for all open positions.

        Args:
            current_prices: Dict mapping pair symbols to current prices.
        """
        triggered_ids = self.risk_manager.check_stop_losses(current_prices)
        for trade_id in triggered_ids:
            trade = self._open_trades.get(trade_id)
            if not trade:
                continue
            price = current_prices.get(trade["pair"], 0.0)
            pnl = self.risk_manager.close_position(trade_id, price)
            self.trade_logger.log_exit(trade_id, price, pnl, reason="stop_loss")
            del self._open_trades[trade_id]
            logger.info("Stop-loss closed %s: pnl=%.4f", trade_id, pnl)
