"""Entry point for the multi-asset paper trading bot.

Loads configuration from environment variables, initializes broker
adapters (Binance for crypto, Alpaca for stocks/ETFs), and starts
the trading session loop.
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from bot.broker_base import BrokerAdapter
from bot.session_manager import SessionConfig, SessionManager
from bot.trade_logger import TradeLogger

logger_mod = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the bot.

    Args:
        level: Log level string (e.g., 'DEBUG', 'INFO', 'WARNING').
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config_from_env() -> SessionConfig:
    """Load all configuration from environment variables.

    Returns:
        SessionConfig with all trading parameters.
    """
    crypto_pairs_raw = os.environ.get("TRADING_PAIRS", "")
    crypto_pairs = [p.strip() for p in crypto_pairs_raw.split(",") if p.strip()]

    stock_symbols_raw = os.environ.get("STOCK_SYMBOLS", "")
    stock_symbols = [s.strip() for s in stock_symbols_raw.split(",") if s.strip()]

    config = SessionConfig(
        pairs=crypto_pairs,
        stock_symbols=stock_symbols,
        timeframe=os.environ.get("TIMEFRAME", "1m"),
        stock_timeframe=os.environ.get("STOCK_TIMEFRAME", "5m"),
        candle_limit=int(os.environ.get("CANDLE_LIMIT", "100")),
        poll_interval_seconds=float(os.environ.get("POLL_INTERVAL", "60")),
        stock_poll_interval_seconds=float(os.environ.get("STOCK_POLL_INTERVAL", "120")),
        starting_capital=float(os.environ.get("STARTING_CAPITAL", "1000.0")),
        max_risk_per_trade_pct=float(os.environ.get("MAX_RISK_PCT", "0.02")),
        max_concurrent_positions=int(os.environ.get("MAX_POSITIONS", "3")),
        daily_hard_stop_pct=float(os.environ.get("DAILY_HARD_STOP_PCT", "-0.10")),
        ema_fast_period=int(os.environ.get("EMA_FAST", "9")),
        ema_slow_period=int(os.environ.get("EMA_SLOW", "21")),
        rsi_period=int(os.environ.get("RSI_PERIOD", "14")),
        atr_period=int(os.environ.get("ATR_PERIOD", "14")),
        adx_period=int(os.environ.get("ADX_PERIOD", "14")),
        bb_period=int(os.environ.get("BB_PERIOD", "20")),
        adx_threshold=float(os.environ.get("ADX_THRESHOLD", "25.0")),
        bb_width_threshold=float(os.environ.get("BB_WIDTH_THRESHOLD", "0.02")),
        skip_market_hours_check=os.environ.get("SKIP_MARKET_HOURS_CHECK", "false").lower()
        in ("true", "1", "yes"),
    )

    return config


async def build_brokers(config: SessionConfig) -> dict[str, BrokerAdapter]:
    """Initialize and connect all configured broker adapters.

    Args:
        config: Session configuration.

    Returns:
        Dict mapping broker names to connected adapter instances.
    """
    brokers: dict[str, BrokerAdapter] = {}

    # Binance for crypto
    if config.pairs:
        binance_key = os.environ.get("BINANCE_API_KEY", "")
        binance_secret = os.environ.get("BINANCE_API_SECRET", "")

        if binance_key and binance_secret:
            from bot.broker_binance import BinanceAdapter

            adapter = BinanceAdapter(
                api_key=binance_key,
                api_secret=binance_secret,
                testnet=os.environ.get("BINANCE_TESTNET", "true").lower()
                in ("true", "1", "yes"),
                rate_limit=os.environ.get("BINANCE_RATE_LIMIT", "true").lower()
                in ("true", "1", "yes"),
            )
            await adapter.connect()
            brokers["binance"] = adapter
            logger_mod.info(
                "Binance adapter ready: %d crypto pairs", len(config.pairs)
            )
        else:
            logger_mod.warning(
                "BINANCE_API_KEY/SECRET not set — skipping crypto pairs: %s",
                config.pairs,
            )
            config.pairs = []

    # Alpaca for stocks/ETFs
    if config.stock_symbols:
        alpaca_key = os.environ.get("ALPACA_API_KEY", "")
        alpaca_secret = os.environ.get("ALPACA_API_SECRET", "")

        if alpaca_key and alpaca_secret:
            from bot.broker_alpaca import AlpacaAdapter

            adapter = AlpacaAdapter(
                api_key=alpaca_key,
                api_secret=alpaca_secret,
                paper=os.environ.get("ALPACA_PAPER", "true").lower()
                in ("true", "1", "yes"),
            )
            await adapter.connect()
            brokers["alpaca"] = adapter
            logger_mod.info(
                "Alpaca adapter ready: %d stock symbols", len(config.stock_symbols)
            )
        else:
            logger_mod.warning(
                "ALPACA_API_KEY/SECRET not set — skipping stock symbols: %s",
                config.stock_symbols,
            )
            config.stock_symbols = []

    return brokers


async def run_bot(config: SessionConfig, log_path: str = "trades.jsonl") -> None:
    """Initialize all components and run the trading session.

    Args:
        config: Session configuration.
        log_path: Path for the trade log file.
    """
    if not config.pairs and not config.stock_symbols:
        logger_mod.critical(
            "No trading symbols configured. Set TRADING_PAIRS and/or STOCK_SYMBOLS."
        )
        sys.exit(1)

    brokers = await build_brokers(config)

    if not brokers:
        logger_mod.critical(
            "No brokers could be initialized. Check API keys."
        )
        sys.exit(1)

    trade_logger = TradeLogger(log_path=log_path)
    session = SessionManager(
        config=config,
        brokers=brokers,
        trade_logger=trade_logger,
    )

    loop = asyncio.get_running_loop()

    def _shutdown(sig: signal.Signals) -> None:
        logger_mod.info("Received %s — shutting down", sig.name)
        session.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig)
        except NotImplementedError:
            pass

    try:
        await session.run()
    finally:
        for name, broker in brokers.items():
            try:
                await broker.disconnect()
            except Exception as exc:
                logger_mod.warning("Error disconnecting %s: %s", name, exc)

        final_stats = session.risk_manager.get_session_stats()
        trade_logger.log_session_summary(final_stats)
        logger_mod.info(
            "Final stats: capital=%.2f, return=%.2f%%, trades=%d",
            final_stats["current_capital"],
            final_stats["session_return_pct"] * 100,
            final_stats["total_trades"],
        )


def main() -> None:
    """CLI entry point for the paper trading bot."""
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    setup_logging(log_level)

    logger_mod.info("Multi-Asset Paper Trading Bot starting up")

    config = load_config_from_env()
    log_path = os.environ.get("TRADE_LOG_PATH", "trades.jsonl")

    logger_mod.info(
        "Config: crypto=%s, stocks=%s, capital=%.0f, risk=%.1f%%, max_pos=%d",
        config.pairs or "(none)",
        config.stock_symbols or "(none)",
        config.starting_capital,
        config.max_risk_per_trade_pct * 100,
        config.max_concurrent_positions,
    )

    try:
        asyncio.run(run_bot(config, log_path))
    except KeyboardInterrupt:
        logger_mod.info("Interrupted by user")
    except Exception as exc:
        logger_mod.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
