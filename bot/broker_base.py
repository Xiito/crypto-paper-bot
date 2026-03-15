"""Abstract base class for all broker adapters.

Defines the unified interface that every broker (Binance, Alpaca, etc.)
must implement. The session manager and strategy engine interact only
with this interface, making asset-class switching transparent.
"""

import abc
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AssetClass(str, Enum):
    """Supported asset classes."""

    CRYPTO = "crypto"
    STOCK = "stock"
    ETF = "etf"


class MarketStatus(str, Enum):
    """Market open/closed status."""

    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"
    UNKNOWN = "unknown"


@dataclass
class OrderResult:
    """Result of an order execution (unified across brokers)."""

    success: bool
    order_id: Optional[str] = None
    pair: Optional[str] = None
    direction: Optional[str] = None
    price: float = 0.0
    quantity: float = 0.0
    timestamp: float = 0.0
    error: Optional[str] = None
    asset_class: Optional[str] = None
    broker: Optional[str] = None


@dataclass
class MarketInfo:
    """Market metadata for a trading symbol."""

    symbol: str
    asset_class: AssetClass
    min_quantity: float = 0.0
    max_quantity: float = 0.0
    quantity_step: float = 0.0
    min_notional: float = 0.0
    price_precision: int = 8
    quantity_precision: int = 8
    tradeable: bool = True
    shortable: bool = True


class BrokerAdapter(abc.ABC):
    """Abstract broker adapter interface.

    Every broker implementation (Binance, Alpaca, etc.) must subclass
    this and implement all abstract methods. The session manager only
    interacts with this interface.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the broker name (e.g., 'binance', 'alpaca')."""

    @property
    @abc.abstractmethod
    def asset_class(self) -> AssetClass:
        """Return the primary asset class this broker handles."""

    @abc.abstractmethod
    async def connect(self) -> None:
        """Initialize the broker connection and load markets."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Close the broker connection gracefully."""

    @abc.abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 100
    ) -> list[list]:
        """Fetch OHLCV candlestick data.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT' or 'AAPL').
            timeframe: Candle timeframe (e.g., '1m', '5m', '1h', '1d').
            limit: Number of candles to fetch.

        Returns:
            List of [timestamp_ms, open, high, low, close, volume].
        """

    @abc.abstractmethod
    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker data for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            Dict with at minimum 'last', 'bid', 'ask' prices.
        """

    @abc.abstractmethod
    async def execute_market_order(
        self, symbol: str, direction: str, quantity: float
    ) -> OrderResult:
        """Execute a market order.

        Args:
            symbol: Trading symbol.
            direction: 'LONG' (buy) or 'SHORT' (sell/short).
            quantity: Order quantity in base units.

        Returns:
            OrderResult with execution details.
        """

    @abc.abstractmethod
    async def get_market_status(self) -> MarketStatus:
        """Check whether the market is currently open for trading.

        Returns:
            MarketStatus enum value.
        """

    @abc.abstractmethod
    async def get_account_info(self) -> dict:
        """Fetch account information (balance, buying power, etc.).

        Returns:
            Dict with account details.
        """

    @abc.abstractmethod
    def get_market_info(self, symbol: str) -> Optional[MarketInfo]:
        """Get market metadata for a symbol (precision, lot sizes, etc.).

        Args:
            symbol: Trading symbol.

        Returns:
            MarketInfo or None if the symbol is not found.
        """

    def normalize_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to the symbol's allowed precision.

        Args:
            symbol: Trading symbol.
            quantity: Raw quantity to normalize.

        Returns:
            Normalized quantity that complies with broker rules.
        """
        info = self.get_market_info(symbol)
        if info is None:
            return quantity
        if info.quantity_step > 0:
            quantity = (quantity // info.quantity_step) * info.quantity_step
        return round(quantity, info.quantity_precision)
