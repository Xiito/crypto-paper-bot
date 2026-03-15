"""Yahoo Finance broker adapter for simulated stock/ETF paper trading.

Uses the yfinance library to fetch real-time market data for US
stocks and ETFs, with fully simulated order execution. No API key
or brokerage account required.

Trades are simulated locally using current market prices — ideal
for paper trading where the goal is strategy testing, not broker
connectivity.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import yfinance as yf

from bot.broker_base import (
    AssetClass,
    BrokerAdapter,
    MarketInfo,
    MarketStatus,
    OrderResult,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_RETRY_DELAY = 2.0

# Thread pool for running synchronous yfinance calls without blocking
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="yfinance")

# US Eastern timezone for market hours
ET = ZoneInfo("America/New_York")

# Timeframe mapping: our standard → yfinance interval strings
TIMEFRAME_MAP = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "1d": "1d",
}

# yfinance period needed for each timeframe (must cover enough candles)
PERIOD_MAP = {
    "1m": "1d",       # 1m data only available for last 7 days
    "2m": "5d",
    "5m": "5d",
    "15m": "5d",
    "30m": "1mo",
    "1h": "1mo",
    "1d": "6mo",
}


class YahooFinanceAdapter(BrokerAdapter):
    """Yahoo Finance adapter for simulated stock/ETF trading.

    Fetches real market data via yfinance and simulates order
    execution using current prices. No API key required.
    """

    def __init__(self) -> None:
        """Initialize the Yahoo Finance adapter."""
        self._tickers: dict[str, yf.Ticker] = {}
        self._ticker_info: dict[str, dict] = {}
        self._connected = False
        self._order_counter = 0

    @property
    def name(self) -> str:
        return "yahoo"

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.STOCK

    async def connect(self) -> None:
        """Mark the adapter as connected.

        No actual connection needed — yfinance uses public endpoints.
        We verify connectivity by fetching a test ticker.
        """
        try:
            loop = asyncio.get_running_loop()
            test = await loop.run_in_executor(
                _executor, lambda: yf.Ticker("SPY").fast_info
            )
            if test is not None:
                self._connected = True
                logger.info(
                    "Yahoo Finance adapter connected (simulated trading mode)"
                )
            else:
                raise ConnectionError("Could not fetch test ticker")
        except Exception as exc:
            raise ConnectionError(
                f"Yahoo Finance connectivity check failed: {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """Clean up resources."""
        self._tickers.clear()
        self._ticker_info.clear()
        self._connected = False
        logger.info("Yahoo Finance adapter disconnected")

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create a cached yfinance Ticker object."""
        if symbol not in self._tickers:
            self._tickers[symbol] = yf.Ticker(symbol)
        return self._tickers[symbol]

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "5m", limit: int = 100
    ) -> list[list]:
        """Fetch OHLCV candles from Yahoo Finance.

        Returns data in the unified format:
        [[timestamp_ms, open, high, low, close, volume], ...]
        """
        yf_interval = TIMEFRAME_MAP.get(timeframe, "5m")
        yf_period = PERIOD_MAP.get(timeframe, "5d")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                loop = asyncio.get_running_loop()
                ticker = self._get_ticker(symbol)

                df = await loop.run_in_executor(
                    _executor,
                    lambda: ticker.history(
                        period=yf_period,
                        interval=yf_interval,
                        auto_adjust=True,
                    ),
                )

                if df is None or df.empty:
                    logger.warning("No data returned for %s (%s)", symbol, timeframe)
                    return []

                # Take the last `limit` candles
                df = df.tail(limit)

                ohlcv = []
                for idx, row in df.iterrows():
                    ts_ms = int(idx.timestamp() * 1000)
                    ohlcv.append([
                        ts_ms,
                        float(row["Open"]),
                        float(row["High"]),
                        float(row["Low"]),
                        float(row["Close"]),
                        float(row["Volume"]),
                    ])

                logger.debug(
                    "Fetched %d candles for %s (%s)", len(ohlcv), symbol, timeframe
                )
                return ohlcv

            except Exception as exc:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Error fetching OHLCV for %s (attempt %d/%d): %s",
                    symbol, attempt, MAX_RETRIES, exc,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Failed to fetch OHLCV for {symbol} after {MAX_RETRIES} attempts"
        )

    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current price data for a symbol."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                loop = asyncio.get_running_loop()
                ticker = self._get_ticker(symbol)

                info = await loop.run_in_executor(
                    _executor,
                    lambda: ticker.fast_info,
                )

                last_price = float(getattr(info, "last_price", 0) or 0)
                bid = float(getattr(info, "last_price", 0) or 0)
                ask = float(getattr(info, "last_price", 0) or 0)

                # Try to get bid/ask from regular info if available
                try:
                    full_info = await loop.run_in_executor(
                        _executor,
                        lambda: ticker.info,
                    )
                    bid = float(full_info.get("bid", last_price) or last_price)
                    ask = float(full_info.get("ask", last_price) or last_price)
                except Exception:
                    pass

                return {
                    "last": last_price,
                    "bid": bid,
                    "ask": ask,
                    "symbol": symbol,
                }

            except Exception as exc:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Error fetching ticker for %s (attempt %d/%d): %s",
                    symbol, attempt, MAX_RETRIES, exc,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Failed to fetch ticker for {symbol} after {MAX_RETRIES} attempts"
        )

    async def execute_market_order(
        self, symbol: str, direction: str, quantity: float
    ) -> OrderResult:
        """Simulate a market order using current Yahoo Finance price.

        Since this is paper trading, no real order is placed.
        The current market price is used as the fill price.
        """
        quantity = self.normalize_quantity(symbol, quantity)

        if quantity <= 0:
            return OrderResult(
                success=False, pair=symbol, direction=direction,
                error="Quantity must be > 0", broker=self.name,
            )

        try:
            ticker = await self.fetch_ticker(symbol)
            price = float(ticker["last"])

            if price <= 0:
                return OrderResult(
                    success=False, pair=symbol, direction=direction,
                    error=f"Invalid price for {symbol}: {price}",
                    broker=self.name,
                )

            self._order_counter += 1
            order_id = f"SIM_{symbol}_{int(time.time())}_{self._order_counter}"

            logger.info(
                "Simulated order: %s %s, qty=%.2f, price=%.2f, id=%s",
                direction, symbol, quantity, price, order_id,
            )

            return OrderResult(
                success=True,
                order_id=order_id,
                pair=symbol,
                direction=direction,
                price=price,
                quantity=quantity,
                timestamp=time.time(),
                asset_class=AssetClass.STOCK.value,
                broker=self.name,
            )

        except Exception as exc:
            logger.error(
                "Failed to simulate order for %s: %s", symbol, exc
            )
            return OrderResult(
                success=False, pair=symbol, direction=direction,
                error=f"Simulation failed: {exc}", broker=self.name,
            )

    async def get_market_status(self) -> MarketStatus:
        """Check if the US stock market is currently open.

        Uses the market hours manager logic to determine status
        based on current Eastern Time.
        """
        now_et = datetime.now(ET)

        # Weekend
        if now_et.weekday() >= 5:
            return MarketStatus.CLOSED

        from datetime import time as dt_time
        current_time = now_et.time()

        if dt_time(9, 30) <= current_time < dt_time(16, 0):
            return MarketStatus.OPEN
        elif dt_time(4, 0) <= current_time < dt_time(9, 30):
            return MarketStatus.PRE_MARKET
        elif dt_time(16, 0) <= current_time < dt_time(20, 0):
            return MarketStatus.AFTER_HOURS
        else:
            return MarketStatus.CLOSED

    async def get_account_info(self) -> dict:
        """Return simulated account info."""
        return {
            "broker": self.name,
            "mode": "simulated",
            "note": "Paper trading with Yahoo Finance data — no real account",
        }

    def get_market_info(self, symbol: str) -> Optional[MarketInfo]:
        """Get market info for a stock/ETF symbol.

        For stocks, quantity precision is 0 (whole shares),
        with a step of 1.
        """
        return MarketInfo(
            symbol=symbol,
            asset_class=AssetClass.STOCK,
            min_quantity=1.0,
            max_quantity=0,
            quantity_step=1.0,
            min_notional=1.0,
            price_precision=2,
            quantity_precision=0,
            tradeable=True,
            shortable=False,  # Simulated mode: long only
        )
