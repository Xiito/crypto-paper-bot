"""Binance broker adapter for crypto trading via CCXT.

Wraps the existing CCXT Binance integration behind the unified
BrokerAdapter interface. Supports testnet and mainnet modes with
retry logic and rate limiting.
"""

import asyncio
import logging
import time
from typing import Optional

import ccxt.async_support as ccxt

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


class BinanceAdapter(BrokerAdapter):
    """Binance exchange adapter for crypto spot trading.

    Connects to Binance mainnet or testnet via CCXT and implements
    the full BrokerAdapter interface for crypto pairs.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        rate_limit: bool = True,
    ) -> None:
        """Initialize the Binance adapter.

        Args:
            api_key: Binance API key.
            api_secret: Binance API secret.
            testnet: Use testnet if True.
            rate_limit: Enable CCXT rate limiting.
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._rate_limit = rate_limit
        self._exchange: Optional[ccxt.binance] = None
        self._markets: dict = {}

    @property
    def name(self) -> str:
        return "binance"

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.CRYPTO

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
        self._markets = self._exchange.markets
        logger.info(
            "Binance %s connected (markets: %d)",
            "Testnet" if self._testnet else "Mainnet",
            len(self._markets),
        )

    async def disconnect(self) -> None:
        """Close the exchange connection."""
        if self._exchange:
            await self._exchange.close()
            logger.info("Binance connection closed")

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 100
    ) -> list[list]:
        """Fetch OHLCV candles from Binance."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                ohlcv = await self._exchange.fetch_ohlcv(
                    symbol, timeframe, limit=limit
                )
                logger.debug(
                    "Fetched %d candles for %s (%s)", len(ohlcv), symbol, timeframe
                )
                return ohlcv
            except ccxt.RateLimitExceeded:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Rate limit fetching OHLCV %s (attempt %d/%d), waiting %.1fs",
                    symbol, attempt, MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
            except ccxt.NetworkError as exc:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Network error fetching OHLCV %s (attempt %d/%d): %s",
                    symbol, attempt, MAX_RETRIES, exc,
                )
                await asyncio.sleep(delay)
            except ccxt.ExchangeError as exc:
                raise RuntimeError(
                    f"Exchange error fetching OHLCV for {symbol}: {exc}"
                ) from exc

        raise RuntimeError(
            f"Failed to fetch OHLCV for {symbol} after {MAX_RETRIES} attempts"
        )

    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker from Binance."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                ticker = await self._exchange.fetch_ticker(symbol)
                return ticker
            except ccxt.RateLimitExceeded:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Rate limit fetching ticker %s (attempt %d/%d)",
                    symbol, attempt, MAX_RETRIES,
                )
                await asyncio.sleep(delay)
            except ccxt.NetworkError as exc:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Network error fetching ticker %s (attempt %d/%d): %s",
                    symbol, attempt, MAX_RETRIES, exc,
                )
                await asyncio.sleep(delay)
            except ccxt.ExchangeError as exc:
                raise RuntimeError(
                    f"Exchange error fetching ticker for {symbol}: {exc}"
                ) from exc

        raise RuntimeError(
            f"Failed to fetch ticker for {symbol} after {MAX_RETRIES} attempts"
        )

    async def execute_market_order(
        self, symbol: str, direction: str, quantity: float
    ) -> OrderResult:
        """Execute a market order on Binance."""
        side = "buy" if direction == "LONG" else "sell"
        quantity = self.normalize_quantity(symbol, quantity)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                order = await self._exchange.create_market_order(
                    symbol, side, quantity
                )
                price = float(
                    order.get("average") or order.get("price") or 0.0
                )
                logger.info(
                    "Binance order: %s %s %s, qty=%.8f, price=%.8f, id=%s",
                    direction, symbol, side, quantity, price, order["id"],
                )
                return OrderResult(
                    success=True,
                    order_id=str(order["id"]),
                    pair=symbol,
                    direction=direction,
                    price=price,
                    quantity=quantity,
                    timestamp=time.time(),
                    asset_class=AssetClass.CRYPTO.value,
                    broker=self.name,
                )
            except ccxt.InsufficientFunds as exc:
                logger.error("Insufficient funds %s %s: %s", direction, symbol, exc)
                return OrderResult(
                    success=False, pair=symbol, direction=direction,
                    error=str(exc), broker=self.name,
                )
            except ccxt.RateLimitExceeded:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Rate limit placing order %s (attempt %d/%d)",
                    symbol, attempt, MAX_RETRIES,
                )
                await asyncio.sleep(delay)
            except ccxt.NetworkError as exc:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Network error placing order %s (attempt %d/%d): %s",
                    symbol, attempt, MAX_RETRIES, exc,
                )
                await asyncio.sleep(delay)
            except ccxt.ExchangeError as exc:
                logger.error("Exchange error placing order %s: %s", symbol, exc)
                return OrderResult(
                    success=False, pair=symbol, direction=direction,
                    error=str(exc), broker=self.name,
                )

        # Fallback: simulate with current ticker
        logger.error(
            "Failed to execute order %s after %d attempts, simulating",
            symbol, MAX_RETRIES,
        )
        return await self._simulate_order(symbol, direction, quantity)

    async def get_market_status(self) -> MarketStatus:
        """Crypto markets are always open."""
        return MarketStatus.OPEN

    async def get_account_info(self) -> dict:
        """Fetch Binance account balances."""
        try:
            balance = await self._exchange.fetch_balance()
            return {
                "broker": self.name,
                "total_usd": float(balance.get("total", {}).get("USDT", 0)),
                "free_usd": float(balance.get("free", {}).get("USDT", 0)),
                "raw": balance,
            }
        except Exception as exc:
            logger.error("Failed to fetch Binance account info: %s", exc)
            return {"broker": self.name, "error": str(exc)}

    def get_market_info(self, symbol: str) -> Optional[MarketInfo]:
        """Get market info for a Binance symbol."""
        market = self._markets.get(symbol)
        if market is None:
            return None
        limits = market.get("limits", {})
        precision = market.get("precision", {})
        return MarketInfo(
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            min_quantity=float(limits.get("amount", {}).get("min", 0) or 0),
            max_quantity=float(limits.get("amount", {}).get("max", 0) or 0),
            quantity_step=float(market.get("precision", {}).get("amount", 8)),
            min_notional=float(limits.get("cost", {}).get("min", 0) or 0),
            price_precision=int(precision.get("price", 8) or 8),
            quantity_precision=int(precision.get("amount", 8) or 8),
            tradeable=market.get("active", True),
            shortable=True,
        )

    async def _simulate_order(
        self, symbol: str, direction: str, quantity: float
    ) -> OrderResult:
        """Simulate an order using current ticker as fallback."""
        try:
            ticker = await self.fetch_ticker(symbol)
            price = float(ticker["last"])
            sim_id = f"SIM_{symbol}_{int(time.time())}"
            logger.info(
                "Simulated Binance order: %s %s, qty=%.8f, price=%.8f",
                direction, symbol, quantity, price,
            )
            return OrderResult(
                success=True,
                order_id=sim_id,
                pair=symbol,
                direction=direction,
                price=price,
                quantity=quantity,
                timestamp=time.time(),
                asset_class=AssetClass.CRYPTO.value,
                broker=self.name,
            )
        except Exception as exc:
            return OrderResult(
                success=False, pair=symbol, direction=direction,
                error=f"Simulation failed: {exc}", broker=self.name,
            )
