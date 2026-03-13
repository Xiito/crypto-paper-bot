"""Entry point for the crypto paper trading bot.

Loads configuration from environment variables, initializes all
components, and starts the trading session loop.
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from bot.order_executor import OrderExecutor
from bot.session_manager import SessionConfig, SessionManager
from bot.trade_logger import TradeLogger


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


def load_config_from_env() -> tuple[str, str, SessionConfig]:
    """Load all configuration from environment variables.

    Returns:
        Tuple of (api_key, api_secret, SessionConfig).

    Raises:
        SystemExit: If required environment variables are missing.
    """
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        logging.critical(
            "BINANCE_API_KEY and BINANCE_API_SECRET must be set in environment variables"
        )
        sys.exit(1)

    pairs_raw = os.environ.get("TRADING_PAIRS", "BTC/USDT,ETH/USDT,BNB/USDT")
    pairs = [p.strip() for p in pairs_raw.split(",") if p.strip()]

    config = SessionConfig(
        pairs=pairs,
        timeframe=os.environ.get("TIMEFRAME", "1m"),
        candle_limit=int(os.environ.get("CANDLE_LIMIT", "100")),
        poll_interval_seconds=float(os.environ.get("POLL_INTERVAL", "60")),
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
    )

    return api_key, api_secret, config


async def run_bot(
    api_key: str,
    api_secret: str,
    config: SessionConfig,
    log_path: str = "trades.jsonl",
) -> None:
    """Initialize all components and run the trading session.

    Args:
        api_key: Binance API key.
        api_secret: Binance API secret.
        config: Session configuration.
        log_path: Path for the trade log file.
    """
    executor = OrderExecutor(
        api_key=api_key,
        api_secret=api_secret,
        testnet=True,
    )
    trade_logger = TradeLogger(log_path=log_path)
    session = SessionManager(
        config=config,
        executor=executor,
        trade_logger=trade_logger,
    )

    loop = asyncio.get_running_loop()

    def _shutdown(sig: signal.Signals) -> None:
        logging.getLogger(__name__).info("Received %s — shutting down", sig.name)
        session.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig)
        except NotImplementedError:
            pass

    await executor.connect()
    try:
        await session.run()
    finally:
        await executor.disconnect()
        final_stats = session.risk_manager.get_session_stats()
        trade_logger.log_session_summary(final_stats)
        logging.getLogger(__name__).info(
            "Final stats: capital=%.2f, return=%.2f%%, trades=%d",
            final_stats["current_capital"],
            final_stats["session_return_pct"] * 100,
            final_stats["total_trades"],
        )


def main() -> None:
    """CLI entry point for the paper trading bot."""
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    setup_logging(log_level)

    logger = logging.getLogger(__name__)
    logger.info("Crypto Paper Trading Bot starting up")

    api_key, api_secret, config = load_config_from_env()

    log_path = os.environ.get("TRADE_LOG_PATH", "trades.jsonl")

    logger.info(
        "Config: pairs=%s, timeframe=%s, capital=%.0f, risk=%.1f%%, max_pos=%d",
        config.pairs,
        config.timeframe,
        config.starting_capital,
        config.max_risk_per_trade_pct * 100,
        config.max_concurrent_positions,
    )

    try:
        asyncio.run(run_bot(api_key, api_secret, config, log_path))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
