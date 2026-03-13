"""EMA/RSI/ATR signal generation engine.

Computes technical indicators from OHLCV data and generates
LONG/SHORT trading signals based on EMA crossover, RSI range
confirmation, and ATR volatility gating.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalDirection(str, Enum):
    """Trade signal direction."""

    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class TradeSignal:
    """A trading signal with full indicator context."""

    pair: str
    direction: SignalDirection
    ema_fast: float
    ema_slow: float
    ema_cross: bool
    rsi: float
    atr: float
    current_price: float
    stop_loss_price: float
    strength: float  # 0.0 to 1.0


class SignalEngine:
    """Technical analysis signal generator using EMA crossover, RSI, and ATR.

    Evaluates live OHLCV data to produce actionable LONG/SHORT signals.
    A valid signal requires all three conditions:
    1. EMA(fast) crosses above/below EMA(slow)
    2. RSI is within the confirmation range
    3. ATR exceeds the minimum volatility threshold
    """

    def __init__(
        self,
        ema_fast_period: int = 9,
        ema_slow_period: int = 21,
        rsi_period: int = 14,
        atr_period: int = 14,
        rsi_long_min: float = 45.0,
        rsi_long_max: float = 65.0,
        rsi_short_min: float = 35.0,
        rsi_short_max: float = 55.0,
    ) -> None:
        """Initialize the signal engine with indicator parameters.

        Args:
            ema_fast_period: Fast EMA period (default 9).
            ema_slow_period: Slow EMA period (default 21).
            rsi_period: RSI period (default 14).
            atr_period: ATR period (default 14).
            rsi_long_min: Minimum RSI for long signals.
            rsi_long_max: Maximum RSI for long signals.
            rsi_short_min: Minimum RSI for short signals.
            rsi_short_max: Maximum RSI for short signals.
        """
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.rsi_long_min = rsi_long_min
        self.rsi_long_max = rsi_long_max
        self.rsi_short_min = rsi_short_min
        self.rsi_short_max = rsi_short_max
        self._prev_ema_state: dict[str, Optional[str]] = {}

    def evaluate(self, pair: str, ohlcv_data: list[list]) -> TradeSignal:
        """Evaluate OHLCV data and generate a trading signal.

        Args:
            pair: Trading pair symbol (e.g., 'BTC/USDT').
            ohlcv_data: List of [timestamp, open, high, low, close, volume] candles.

        Returns:
            A TradeSignal with the computed direction and indicator values.
        """
        df = self._build_dataframe(ohlcv_data)
        df = self._compute_ema(df)
        df = self._compute_rsi(df)
        df = self._compute_atr(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        ema_fast = float(latest["ema_fast"])
        ema_slow = float(latest["ema_slow"])
        rsi = float(latest["rsi"])
        atr = float(latest["atr"])
        current_price = float(latest["close"])

        bullish_cross = self._detect_cross(prev, latest, "bullish")
        bearish_cross = self._detect_cross(prev, latest, "bearish")

        atr_sufficient = self._check_atr_volatility(df)

        direction = SignalDirection.NONE
        ema_cross = False
        stop_loss_price = current_price
        strength = 0.0

        if bullish_cross and self.rsi_long_min <= rsi <= self.rsi_long_max and atr_sufficient:
            direction = SignalDirection.LONG
            ema_cross = True
            stop_loss_price = current_price - (atr * 1.5)
            strength = self._calculate_strength(rsi, self.rsi_long_min, self.rsi_long_max, atr, df)
            logger.info(
                "LONG signal for %s: price=%.8f, EMA9=%.8f, EMA21=%.8f, RSI=%.2f, ATR=%.8f",
                pair, current_price, ema_fast, ema_slow, rsi, atr,
            )

        elif bearish_cross and self.rsi_short_min <= rsi <= self.rsi_short_max and atr_sufficient:
            direction = SignalDirection.SHORT
            ema_cross = True
            stop_loss_price = current_price + (atr * 1.5)
            strength = self._calculate_strength(rsi, self.rsi_short_min, self.rsi_short_max, atr, df)
            logger.info(
                "SHORT signal for %s: price=%.8f, EMA9=%.8f, EMA21=%.8f, RSI=%.2f, ATR=%.8f",
                pair, current_price, ema_fast, ema_slow, rsi, atr,
            )

        prev_state = "above" if prev["ema_fast"] > prev["ema_slow"] else "below"
        self._prev_ema_state[pair] = prev_state

        return TradeSignal(
            pair=pair,
            direction=direction,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_cross=ema_cross,
            rsi=rsi,
            atr=atr,
            current_price=current_price,
            stop_loss_price=stop_loss_price,
            strength=strength,
        )

    def _build_dataframe(self, ohlcv_data: list[list]) -> pd.DataFrame:
        """Convert raw OHLCV data into a pandas DataFrame.

        Args:
            ohlcv_data: List of [timestamp, open, high, low, close, volume] candles.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df

    def _compute_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute fast and slow Exponential Moving Averages.

        Args:
            df: DataFrame with a 'close' column.

        Returns:
            DataFrame with added 'ema_fast' and 'ema_slow' columns.
        """
        df["ema_fast"] = df["close"].ewm(span=self.ema_fast_period, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.ema_slow_period, adjust=False).mean()
        return df

    def _compute_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the Relative Strength Index.

        Uses the standard Wilder's smoothing method (exponential moving average
        of gains and losses).

        Args:
            df: DataFrame with a 'close' column.

        Returns:
            DataFrame with an added 'rsi' column.
        """
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        df["rsi"] = df["rsi"].fillna(50.0)
        return df

    def _compute_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the Average True Range.

        ATR measures market volatility using high-low range and gaps.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns.

        Returns:
            DataFrame with an added 'atr' column.
        """
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.ewm(span=self.atr_period, adjust=False).mean()
        return df

    def _detect_cross(self, prev: pd.Series, current: pd.Series, cross_type: str) -> bool:
        """Detect an EMA crossover event.

        Args:
            prev: Previous candle data.
            current: Current candle data.
            cross_type: 'bullish' (fast crosses above slow) or 'bearish' (fast crosses below slow).

        Returns:
            True if the specified crossover occurred.
        """
        if cross_type == "bullish":
            return prev["ema_fast"] <= prev["ema_slow"] and current["ema_fast"] > current["ema_slow"]
        elif cross_type == "bearish":
            return prev["ema_fast"] >= prev["ema_slow"] and current["ema_fast"] < current["ema_slow"]
        return False

    def _check_atr_volatility(self, df: pd.DataFrame) -> bool:
        """Check if current ATR indicates sufficient volatility for trading.

        Uses the 25th percentile of recent ATR as the minimum threshold.
        Markets with very low volatility produce unreliable signals.

        Args:
            df: DataFrame with an 'atr' column.

        Returns:
            True if current ATR exceeds the volatility floor.
        """
        atr_values = df["atr"].dropna()
        if len(atr_values) < 2:
            return False
        threshold = atr_values.quantile(0.25)
        current_atr = atr_values.iloc[-1]
        return current_atr > threshold

    def _calculate_strength(
        self,
        rsi: float,
        rsi_min: float,
        rsi_max: float,
        atr: float,
        df: pd.DataFrame,
    ) -> float:
        """Calculate signal strength from 0.0 to 1.0.

        Signal strength is derived from how centered the RSI is within
        its valid range and how high the ATR percentile rank is.

        Args:
            rsi: Current RSI value.
            rsi_min: Minimum RSI for this signal type.
            rsi_max: Maximum RSI for this signal type.
            atr: Current ATR value.
            df: Full DataFrame for percentile calculation.

        Returns:
            Signal strength between 0.0 and 1.0.
        """
        rsi_mid = (rsi_min + rsi_max) / 2.0
        rsi_range = (rsi_max - rsi_min) / 2.0
        rsi_score = max(0.0, 1.0 - abs(rsi - rsi_mid) / rsi_range)

        atr_values = df["atr"].dropna()
        if len(atr_values) > 1:
            atr_percentile = (atr_values < atr).sum() / len(atr_values)
        else:
            atr_percentile = 0.5

        strength = (rsi_score * 0.5) + (atr_percentile * 0.5)
        return round(min(1.0, max(0.0, strength)), 3)
