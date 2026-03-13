"""Binance Testnet order executor using CCXT async.

Handles paper trade order placement, position monitoring, and
OHLCV data fetching with retry logic and rate limiting.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_RETRY_DELAY = 2.0


@dataclass
class OrderResult:
    """Result of an order execution."""

    success: bool
    order_id: Optional[str] = None
    pair: Optional[str] = None
    direction: Optional[str] = None
    price: float = 0.0
    quantity: float = 0.0
    timestamp: float = 0.0
    error: Optional[str] = None


class OrderExecutor:
    """Async order executor for Binance Testnet via CCXT.

    Manages the exchange connection, fetches OHLCV data for signal
    computation, and executes paper trades with exponential backoff
    retry on failures.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        rate_limit: bool = True,
    ) -> None:
        """Initialize the order executor.

        Args:
            api_key: Binance API key.
            api_secret: Binance API secret.
            testnet: Whether to use the testnet (default True).
            rate_limit: Whether to enable CCXT rate limiting.
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._rate_limit = rate_limit
        self._exchange: Optional[ccxt.binance] = None

    async def connect(self) -> None:
        """Initialize the CCXT Binance exchange connection."""
        self._exchange = ccxt.binance(
            {
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "sandbox": self._testnet,
                "enableRateLimit": self._rate_limit,
                "options": {
                    "defaultType": "spot",
                    "adjustForTimeDifference": True,
                },
            }
        )
        if self._testnet:
            self._exchange.set_sandbox_mode(True)

        await self._exchange.load_markets()
        logger.info(
            "Connected to Binance %s (markets loaded: %d)",
            "Testnet" if self._testnet else "Mainnet",
            len(self._exchange.markets),
        )

    async def disconnect(self) -> None:
        """Close the exchange connection."""
        if self._exchange:
            await self._exchange.close()
            logger.info("Binance exchange connection closed")

    async def fetch_ohlcv(self, pair: str, timeframe: str = "1m", limit: int = 100) -> list[list]:
        """Fetch OHLCV (candlestick) data from Binance.

        Args:
            pair: Trading pair symbol (e.g., 'BTC/USDT').
            timeframe: Candle timeframe (e.g., '1m', '5m', '1h').
            limit: Number of candles to fetch.

        Returns:
            List of [timestamp, open, high, low, close, volume] candles.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                ohlcv = await self._exchange.fetch_ohlcv(pair, timeframe, limit=limit)
                logger.debug("Fetched %d candles for %s (%s)", len(ohlcv), pair, timeframe)
                return ohlcv
            except ccxt.RateLimitExceeded as exc:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning("Rate limit hit fetching OHLCV for %s (attempt %d/%d), waiting %.1fs", pair, attempt, MAX_RETRIES, delay)
                await asyncio.sleep(delay)
            except ccxt.NetworkError as exc:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning("Network error fetching OHLCV for %s (attempt %d/%d): %s", pair, attempt, MAX_RETRIES, exc)
                await asyncio.sleep(delay)
            except ccxt.ExchangeError as exc:
                logger.error("Exchange error fetching OHLCV for %s: %s", pair, exc)
                raise RuntimeError(f"Exchange error fetching OHLCV for {pair}: {exc}") from exc

        raise RuntimeError(f"Failed to fetch OHLCV for {pair} after {MAX_RETRIES} attempts")

    async def fetch_ticker(self, pair: str) -> dict:
        """Fetch the current ticker for a trading pair.

        Args:
            pair: Trading pair symbol.

        Returns:
            Ticker dict with 'last', 'bid', 'ask' prices.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                ticker = await self._exchange.fetch_ticker(pair)
                return ticker
            except ccxt.RateLimitExceeded:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning("Rate limit hit fetching ticker for %s (attempt %d/%d)", pair, attempt, MAX_RETRIES)
                await asyncio.sleep(delay)
            except ccxt.NetworkError as exc:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning("Network error fetching ticker for %s (attempt %d/%d): %s", pair, attempt, MAX_RETRIES, exc)
                await asyncio.sleep(delay)
            except ccxt.ExchangeError as exc:
                raise RuntimeError(f"Exchange error fetching ticker for {pair}: {exc}") from exc

        raise RuntimeError(f"Failed to fetch ticker for {pair} after {MAX_RETRIES} attempts")

    async def execute_market_order(
        self,
        pair: str,
        direction: str,
        quantity: float,
    ) -> OrderResult:
        """Execute a market order on Binance Testnet.

        For paper trading, this places a real order on the testnet
        to simulate realistic execution. In the event of testnet issues,
        it falls back to simulated execution using the current ticker.

        Args:
            pair: Trading pair symbol.
            direction: 'LONG' (buy) or 'SHORT' (sell).
            quantity: Order quantity in base currency.

        Returns:
            OrderResult with execution details.
        """
        side = "buy" if direction == "LONG" else "sell"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                order = await self._exchange.create_market_order(pair, side, quantity)
                price = float(order.get("average") or order.get("price") or 0.0)
                logger.info(
                    "Order executed: %s %s %s, qty=%.8f, price=%.8f, id=%s",
                    direction, pair, side, quantity, price, order["id"],
                )
                return OrderResult(
                    success=True,
                    order_id=str(order["id"]),
                    pair=pair,
                    direction=direction,
                    price=price,
                    quantity=quantity,
                    timestamp=time.time(),
                )
            except ccxt.InsufficientFunds as exc:
                logger.error("Insufficient funds for %s %s: %s", direction, pair, exc)
                return OrderResult(success=False, pair=pair, direction=direction, error=str(exc))
            except ccxt.RateLimitExceeded:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning("Rate limit placing order for %s (attempt %d/%d)", pair, attempt, MAX_RETRIES)
                await asyncio.sleep(delay)
            except ccxt.NetworkError as exc:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning("Network error placing order for %s (attempt %d/%d): %s", pair, attempt, MAX_RETRIES, exc)
                await asyncio.sleep(delay)
            except ccxt.ExchangeError as exc:
                logger.error("Exchange error placing order for %s: %s", pair, exc)
                return OrderResult(success=False, pair=pair, direction=direction, error=str(exc))

        logger.error("Failed to execute order for %s after %d attempts, simulating", pair, MAX_RETRIES)
        return await self._simulate_order(pair, direction, quantity)

    async def _simulate_order(
        self,
        pair: str,
        direction: str,
        quantity: float,
    ) -> OrderResult:
        """Simulate an order execution using current ticker price.

        Used as a fallback when testnet is unavailable.

        Args:
            pair: Trading pair symbol.
            direction: 'LONG' or 'SHORT'.
            quantity: Order quantity.

        Returns:
            Simulated OrderResult.
        """
        try:
            ticker = await self.fetch_ticker(pair)
            price = float(ticker["last"])
            sim_id = f"SIM_{pair}_{int(time.time())}"
            logger.info("Simulated order: %s %s, qty=%.8f, price=%.8f", direction, pair, quantity, price)
            return OrderResult(
                success=True,
                order_id=sim_id,
                pair=pair,
                direction=direction,
                price=price,
                quantity=quantity,
                timestamp=time.time(),
            )
        except Exception as exc:
            return OrderResult(success=False, pair=pair, direction=direction, error=f"Simulation failed: {exc}")
