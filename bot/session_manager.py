"""Multi-asset trading session lifecycle manager.

Orchestrates the full trading loop across multiple brokers:
signal generation, regime filtering, risk validation, order
execution, and position monitoring — for both crypto and stocks/ETFs.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from bot.broker_base import AssetClass, BrokerAdapter, MarketStatus, OrderResult
from bot.market_hours import MarketHoursManager
from bot.regime_classifier import RegimeClassifier, MarketRegime
from bot.risk_manager import RiskManager
from bot.strategy import SignalEngine, SignalDirection
from bot.trade_logger import TradeLogger

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Configuration for a trading session."""

    pairs: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    stock_symbols: list[str] = field(default_factory=list)
    timeframe: str = "1m"
    stock_timeframe: str = "5m"
    candle_limit: int = 100
    poll_interval_seconds: float = 60.0
    stock_poll_interval_seconds: float = 120.0
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
    skip_market_hours_check: bool = False


class SessionManager:
    """Manages the full lifecycle of a multi-asset paper trading session.

    Coordinates all bot components — signal engine, regime classifier,
    risk manager, broker adapters, market hours, and trade logger —
    into a continuous polling loop that handles both crypto and stocks.
    """

    def __init__(
        self,
        config: SessionConfig,
        brokers: dict[str, BrokerAdapter],
        trade_logger: TradeLogger,
    ) -> None:
        """Initialize the session manager.

        Args:
            config: Session configuration parameters.
            brokers: Dict mapping broker names to adapter instances.
            trade_logger: Initialized TradeLogger instance.
        """
        self.config = config
        self.brokers = brokers
        self.trade_logger = trade_logger
        self.market_hours = MarketHoursManager()

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

        # Build the symbol-to-broker mapping
        self._symbol_broker: dict[str, str] = {}
        for pair in config.pairs:
            self._symbol_broker[pair] = "binance"
        for sym in config.stock_symbols:
            self._symbol_broker[sym] = "yahoo"

    def _get_broker(self, symbol: str) -> Optional[BrokerAdapter]:
        """Get the broker adapter for a given symbol."""
        broker_name = self._symbol_broker.get(symbol)
        if broker_name is None:
            return None
        return self.brokers.get(broker_name)

    async def run(self) -> None:
        """Start the multi-asset trading session loop.

        Runs crypto and stock loops as concurrent tasks.
        """
        self._running = True

        crypto_pairs = [
            s for s, b in self._symbol_broker.items() if b == "binance"
        ]
        stock_symbols = [
            s for s, b in self._symbol_broker.items() if b == "yahoo"
        ]

        logger.info(
            "Session started: crypto=%s, stocks=%s",
            crypto_pairs or "(none)",
            stock_symbols or "(none)",
        )

        tasks = []
        if crypto_pairs and "binance" in self.brokers:
            tasks.append(
                asyncio.create_task(
                    self._run_asset_loop(
                        symbols=crypto_pairs,
                        broker_name="binance",
                        asset_class=AssetClass.CRYPTO,
                        timeframe=self.config.timeframe,
                        poll_interval=self.config.poll_interval_seconds,
                    ),
                    name="crypto_loop",
                )
            )

        if stock_symbols and "yahoo" in self.brokers:
            tasks.append(
                asyncio.create_task(
                    self._run_asset_loop(
                        symbols=stock_symbols,
                        broker_name="yahoo",
                        asset_class=AssetClass.STOCK,
                        timeframe=self.config.stock_timeframe,
                        poll_interval=self.config.stock_poll_interval_seconds,
                    ),
                    name="stock_loop",
                )
            )

        if not tasks:
            logger.error("No trading loops to run — check broker config")
            return

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Session cancelled")
        finally:
            self._running = False
            logger.info("Session ended")

    def stop(self) -> None:
        """Signal the session loop to stop after the current cycle."""
        self._running = False
        logger.info("Stop signal sent to session manager")

    async def _run_asset_loop(
        self,
        symbols: list[str],
        broker_name: str,
        asset_class: AssetClass,
        timeframe: str,
        poll_interval: float,
    ) -> None:
        """Run the trading loop for one asset class.

        Args:
            symbols: List of symbols to trade.
            broker_name: Name of the broker adapter to use.
            asset_class: The asset class being traded.
            timeframe: OHLCV timeframe.
            poll_interval: Seconds between cycles.
        """
        broker = self.brokers[broker_name]

        while self._running:
            if self.risk_manager.is_hard_stop_triggered:
                logger.critical("Hard stop active — halting %s loop", asset_class.value)
                break

            # Check market hours
            if not self.config.skip_market_hours_check:
                market_status = self.market_hours.is_market_open(asset_class)
                if market_status == MarketStatus.CLOSED:
                    await self.market_hours.wait_for_market_open(asset_class)
                    continue

            # Fetch current prices for open positions
            current_prices = await self._fetch_current_prices(broker, symbols)
            self._monitor_stop_losses(current_prices)
            self.risk_manager.update_unrealized_pnl(current_prices)

            # Process each symbol
            for symbol in symbols:
                if not self._running:
                    break
                await self._process_symbol(
                    symbol=symbol,
                    broker=broker,
                    asset_class=asset_class,
                    timeframe=timeframe,
                )

            stats = self.risk_manager.get_session_stats()
            logger.info(
                "[%s] Cycle complete | capital=%.2f | return=%.2f%% | open=%d | trades=%d",
                asset_class.value,
                stats["current_capital"],
                stats["session_return_pct"] * 100,
                stats["open_positions"],
                stats["total_trades"],
            )

            await asyncio.sleep(poll_interval)

    async def _process_symbol(
        self,
        symbol: str,
        broker: BrokerAdapter,
        asset_class: AssetClass,
        timeframe: str,
    ) -> None:
        """Process a single symbol: fetch data, classify, signal, execute.

        Args:
            symbol: Trading symbol to process.
            broker: Broker adapter to use.
            asset_class: Asset class of this symbol.
            timeframe: OHLCV timeframe.
        """
        try:
            ohlcv = await broker.fetch_ohlcv(
                symbol, timeframe, self.config.candle_limit
            )

            if not ohlcv or len(ohlcv) < 30:
                logger.debug(
                    "Insufficient data for %s (%d candles)", symbol, len(ohlcv) if ohlcv else 0
                )
                return

            regime = self.regime_classifier.classify(ohlcv)
            if not regime.is_tradeable:
                logger.debug(
                    "Skipping %s — regime: %s (ADX=%.1f)",
                    symbol, regime.regime.value, regime.adx,
                )
                return

            signal = self.signal_engine.evaluate(symbol, ohlcv)
            if signal.direction == SignalDirection.NONE:
                return

            # For stocks: check if shortable before allowing SHORT signals
            if asset_class in (AssetClass.STOCK, AssetClass.ETF):
                if signal.direction == SignalDirection.SHORT:
                    info = broker.get_market_info(symbol)
                    if info and not info.shortable:
                        logger.debug(
                            "Skipping SHORT for %s — not shortable", symbol
                        )
                        return

            size_result = self.risk_manager.validate_trade(signal)
            if not size_result.approved:
                logger.debug(
                    "Trade rejected for %s: %s", symbol, size_result.rejection_reason
                )
                return

            # Normalize quantity for stocks (whole shares)
            quantity = size_result.quantity
            if asset_class in (AssetClass.STOCK, AssetClass.ETF):
                quantity = max(1.0, float(int(quantity)))

            order = await broker.execute_market_order(
                symbol, signal.direction.value, quantity
            )
            if not order.success:
                logger.error("Order failed for %s: %s", symbol, order.error)
                return

            trade_id = order.order_id or f"TRADE_{symbol}_{int(time.time())}"
            self.risk_manager.register_open_position(
                trade_id=trade_id,
                pair=symbol,
                direction=signal.direction.value,
                entry_price=order.price,
                quantity=order.quantity,
                stop_loss_price=signal.stop_loss_price,
            )
            self._open_trades[trade_id] = {
                "pair": symbol,
                "direction": signal.direction.value,
                "entry_price": order.price,
                "quantity": order.quantity,
                "stop_loss_price": signal.stop_loss_price,
                "asset_class": asset_class.value,
                "broker": broker.name,
            }
            self.trade_logger.log_entry(trade_id, signal, size_result, order)

        except Exception as exc:
            logger.exception("Error processing %s: %s", symbol, exc)

    async def _fetch_current_prices(
        self, broker: BrokerAdapter, symbols: list[str]
    ) -> dict[str, float]:
        """Fetch current prices for symbols with open positions.

        Args:
            broker: Broker adapter to fetch from.
            symbols: List of symbols this broker handles.

        Returns:
            Dict mapping symbols to current prices.
        """
        prices = {}
        open_pairs = {
            v["pair"]
            for v in self._open_trades.values()
            if v.get("broker") == broker.name
        }
        for symbol in open_pairs:
            try:
                ticker = await broker.fetch_ticker(symbol)
                prices[symbol] = float(ticker["last"])
            except Exception as exc:
                logger.warning("Failed to fetch price for %s: %s", symbol, exc)
        return prices

    def _monitor_stop_losses(self, current_prices: dict[str, float]) -> None:
        """Check and execute stop-losses for all open positions.

        Args:
            current_prices: Dict mapping symbols to current prices.
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
