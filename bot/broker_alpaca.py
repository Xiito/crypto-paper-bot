"""Alpaca broker adapter for stocks and ETFs paper trading.

Uses the Alpaca Markets REST API for paper trading US equities.
Implements the unified BrokerAdapter interface so the session
manager can trade stocks/ETFs identically to crypto.

Alpaca Paper Trading API docs: https://docs.alpaca.markets/
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp

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

ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
ALPACA_DATA_BASE = "https://data.alpaca.markets"

# Timeframe mapping: our standard timeframes → Alpaca bar timeframes
TIMEFRAME_MAP = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "1d": "1Day",
}


class AlpacaAdapter(BrokerAdapter):
    """Alpaca broker adapter for US stocks and ETFs.

    Connects to Alpaca's paper trading API and data API to
    provide OHLCV data, order execution, and market status
    checks for US equities.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
    ) -> None:
        """Initialize the Alpaca adapter.

        Args:
            api_key: Alpaca API key ID.
            api_secret: Alpaca API secret key.
            paper: Use paper trading if True (default).
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._paper = paper
        self._base_url = ALPACA_PAPER_BASE if paper else "https://api.alpaca.markets"
        self._data_url = ALPACA_DATA_BASE
        self._session: Optional[aiohttp.ClientSession] = None
        self._assets: dict[str, dict] = {}

    @property
    def name(self) -> str:
        return "alpaca"

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.STOCK

    def _headers(self) -> dict[str, str]:
        """Build authentication headers for Alpaca API."""
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
            "Content-Type": "application/json",
        }

    async def connect(self) -> None:
        """Initialize the HTTP session and load tradeable assets."""
        self._session = aiohttp.ClientSession(headers=self._headers())

        # Verify credentials with account endpoint
        async with self._session.get(f"{self._base_url}/v2/account") as resp:
            if resp.status != 200:
                body = await resp.text()
                await self._session.close()
                raise ConnectionError(
                    f"Alpaca authentication failed ({resp.status}): {body}"
                )
            account = await resp.json()
            logger.info(
                "Alpaca %s connected: account=%s, equity=$%s, buying_power=$%s",
                "Paper" if self._paper else "Live",
                account.get("account_number", "?"),
                account.get("equity", "0"),
                account.get("buying_power", "0"),
            )

        # Load tradeable assets (US equities)
        async with self._session.get(
            f"{self._base_url}/v2/assets",
            params={"status": "active", "asset_class": "us_equity"},
        ) as resp:
            if resp.status == 200:
                assets_list = await resp.json()
                self._assets = {a["symbol"]: a for a in assets_list}
                logger.info("Loaded %d Alpaca assets", len(self._assets))
            else:
                logger.warning("Failed to load Alpaca assets: %d", resp.status)

    async def disconnect(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            logger.info("Alpaca connection closed")

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 100
    ) -> list[list]:
        """Fetch OHLCV bars from Alpaca Data API.

        Returns data in the same format as crypto:
        [[timestamp_ms, open, high, low, close, volume], ...]
        """
        alpaca_tf = TIMEFRAME_MAP.get(timeframe, "1Min")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                url = f"{self._data_url}/v2/stocks/{symbol}/bars"
                params = {
                    "timeframe": alpaca_tf,
                    "limit": limit,
                    "adjustment": "split",
                    "feed": "iex",
                    "sort": "asc",
                }

                async with self._session.get(url, params=params) as resp:
                    if resp.status == 422:
                        # Try sip feed if iex fails
                        params["feed"] = "sip"
                        async with self._session.get(url, params=params) as resp2:
                            if resp2.status != 200:
                                body = await resp2.text()
                                raise RuntimeError(
                                    f"Alpaca bars error ({resp2.status}): {body}"
                                )
                            data = await resp2.json()
                    elif resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(
                            f"Alpaca bars error ({resp.status}): {body}"
                        )
                    else:
                        data = await resp.json()

                bars = data.get("bars") or []
                ohlcv = []
                for bar in bars:
                    ts = bar["t"]
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        ts_ms = int(dt.timestamp() * 1000)
                    else:
                        ts_ms = int(ts)
                    ohlcv.append([
                        ts_ms,
                        float(bar["o"]),
                        float(bar["h"]),
                        float(bar["l"]),
                        float(bar["c"]),
                        float(bar["v"]),
                    ])

                logger.debug(
                    "Fetched %d bars for %s (%s)", len(ohlcv), symbol, timeframe
                )
                return ohlcv

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Network error fetching bars %s (attempt %d/%d): %s",
                    symbol, attempt, MAX_RETRIES, exc,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Failed to fetch OHLCV for {symbol} after {MAX_RETRIES} attempts"
        )

    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch latest trade/quote for a symbol."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Fetch latest trade
                trade_url = f"{self._data_url}/v2/stocks/{symbol}/trades/latest"
                quote_url = f"{self._data_url}/v2/stocks/{symbol}/quotes/latest"

                async with self._session.get(
                    trade_url, params={"feed": "iex"}
                ) as resp:
                    if resp.status == 200:
                        trade_data = await resp.json()
                        last_price = float(trade_data.get("trade", {}).get("p", 0))
                    else:
                        last_price = 0.0

                async with self._session.get(
                    quote_url, params={"feed": "iex"}
                ) as resp:
                    if resp.status == 200:
                        quote_data = await resp.json()
                        quote = quote_data.get("quote", {})
                        bid = float(quote.get("bp", 0))
                        ask = float(quote.get("ap", 0))
                    else:
                        bid, ask = last_price, last_price

                return {
                    "last": last_price or ((bid + ask) / 2 if bid and ask else 0),
                    "bid": bid,
                    "ask": ask,
                    "symbol": symbol,
                }

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Network error fetching ticker %s (attempt %d/%d): %s",
                    symbol, attempt, MAX_RETRIES, exc,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Failed to fetch ticker for {symbol} after {MAX_RETRIES} attempts"
        )

    async def execute_market_order(
        self, symbol: str, direction: str, quantity: float
    ) -> OrderResult:
        """Execute a market order on Alpaca.

        For paper trading, Alpaca simulates real market execution
        including partial fills and realistic slippage.
        """
        side = "buy" if direction == "LONG" else "sell"
        quantity = self.normalize_quantity(symbol, quantity)

        if quantity <= 0:
            return OrderResult(
                success=False, pair=symbol, direction=direction,
                error="Quantity must be > 0", broker=self.name,
            )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                order_payload = {
                    "symbol": symbol,
                    "qty": str(quantity),
                    "side": side,
                    "type": "market",
                    "time_in_force": "day",
                }

                async with self._session.post(
                    f"{self._base_url}/v2/orders",
                    json=order_payload,
                ) as resp:
                    if resp.status in (200, 201):
                        order = await resp.json()
                        filled_price = float(order.get("filled_avg_price") or 0)

                        # If not filled yet, wait briefly and check
                        if filled_price == 0 and order.get("status") in (
                            "new", "accepted", "pending_new",
                        ):
                            filled_price = await self._wait_for_fill(
                                order["id"], symbol
                            )

                        logger.info(
                            "Alpaca order: %s %s %s, qty=%s, price=%.2f, id=%s",
                            direction, symbol, side, quantity,
                            filled_price, order["id"],
                        )
                        return OrderResult(
                            success=True,
                            order_id=str(order["id"]),
                            pair=symbol,
                            direction=direction,
                            price=filled_price,
                            quantity=float(order.get("filled_qty") or quantity),
                            timestamp=time.time(),
                            asset_class=AssetClass.STOCK.value,
                            broker=self.name,
                        )
                    elif resp.status == 403:
                        body = await resp.json()
                        msg = body.get("message", "Forbidden")
                        logger.error("Alpaca order rejected for %s: %s", symbol, msg)
                        return OrderResult(
                            success=False, pair=symbol, direction=direction,
                            error=msg, broker=self.name,
                        )
                    elif resp.status == 422:
                        body = await resp.json()
                        msg = body.get("message", "Unprocessable")
                        logger.error("Alpaca validation error for %s: %s", symbol, msg)
                        return OrderResult(
                            success=False, pair=symbol, direction=direction,
                            error=msg, broker=self.name,
                        )
                    else:
                        body = await resp.text()
                        raise RuntimeError(
                            f"Alpaca order error ({resp.status}): {body}"
                        )

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Network error placing order %s (attempt %d/%d): %s",
                    symbol, attempt, MAX_RETRIES, exc,
                )
                await asyncio.sleep(delay)

        # Fallback: simulate with ticker price
        logger.error(
            "Failed to execute Alpaca order %s after %d attempts, simulating",
            symbol, MAX_RETRIES,
        )
        return await self._simulate_order(symbol, direction, quantity)

    async def get_market_status(self) -> MarketStatus:
        """Check if the US stock market is currently open."""
        try:
            async with self._session.get(
                f"{self._base_url}/v2/clock"
            ) as resp:
                if resp.status != 200:
                    return MarketStatus.UNKNOWN
                clock = await resp.json()
                if clock.get("is_open"):
                    return MarketStatus.OPEN
                return MarketStatus.CLOSED
        except Exception as exc:
            logger.warning("Failed to check Alpaca market status: %s", exc)
            return MarketStatus.UNKNOWN

    async def get_account_info(self) -> dict:
        """Fetch Alpaca account information."""
        try:
            async with self._session.get(
                f"{self._base_url}/v2/account"
            ) as resp:
                if resp.status != 200:
                    return {"broker": self.name, "error": f"HTTP {resp.status}"}
                account = await resp.json()
                return {
                    "broker": self.name,
                    "total_usd": float(account.get("equity", 0)),
                    "free_usd": float(account.get("buying_power", 0)),
                    "cash": float(account.get("cash", 0)),
                    "portfolio_value": float(account.get("portfolio_value", 0)),
                    "raw": account,
                }
        except Exception as exc:
            logger.error("Failed to fetch Alpaca account info: %s", exc)
            return {"broker": self.name, "error": str(exc)}

    def get_market_info(self, symbol: str) -> Optional[MarketInfo]:
        """Get market info for an Alpaca symbol."""
        asset = self._assets.get(symbol)
        if asset is None:
            return None

        is_etf = asset.get("class", "") == "us_equity" and asset.get(
            "exchange", ""
        ) in ("ARCA", "BATS")

        return MarketInfo(
            symbol=symbol,
            asset_class=AssetClass.ETF if is_etf else AssetClass.STOCK,
            min_quantity=1.0,
            max_quantity=0,  # No upper limit
            quantity_step=1.0,  # Whole shares (fractional requires flag)
            min_notional=1.0,
            price_precision=2,
            quantity_precision=0,  # Whole shares
            tradeable=asset.get("tradable", True),
            shortable=asset.get("shortable", False),
        )

    async def _wait_for_fill(
        self, order_id: str, symbol: str, max_wait: float = 10.0
    ) -> float:
        """Poll order status until filled or timeout.

        Args:
            order_id: Alpaca order ID.
            symbol: Symbol for logging.
            max_wait: Maximum seconds to wait for fill.

        Returns:
            Filled average price, or ticker last price as fallback.
        """
        start = time.time()
        while (time.time() - start) < max_wait:
            await asyncio.sleep(0.5)
            try:
                async with self._session.get(
                    f"{self._base_url}/v2/orders/{order_id}"
                ) as resp:
                    if resp.status == 200:
                        order = await resp.json()
                        if order.get("status") == "filled":
                            return float(order.get("filled_avg_price", 0))
                        if order.get("status") in ("canceled", "expired", "rejected"):
                            break
            except Exception:
                pass

        # Fallback: use latest ticker price
        try:
            ticker = await self.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception:
            return 0.0

    async def _simulate_order(
        self, symbol: str, direction: str, quantity: float
    ) -> OrderResult:
        """Simulate an order using current ticker as fallback."""
        try:
            ticker = await self.fetch_ticker(symbol)
            price = float(ticker["last"])
            sim_id = f"SIM_{symbol}_{int(time.time())}"
            logger.info(
                "Simulated Alpaca order: %s %s, qty=%.2f, price=%.2f",
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
                asset_class=AssetClass.STOCK.value,
                broker=self.name,
            )
        except Exception as exc:
            return OrderResult(
                success=False, pair=symbol, direction=direction,
                error=f"Simulation failed: {exc}", broker=self.name,
            )
