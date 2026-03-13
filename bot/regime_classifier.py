"""Market regime classifier using ADX and Bollinger Band width.

Classifies current market conditions as TRENDING or RANGING to
filter out low-quality signals in choppy, sideways markets.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classification."""

    TRENDING = "TRENDING"
    RANGING = "RANGING"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeResult:
    """Result of a market regime classification."""

    regime: MarketRegime
    adx: float
    bb_width: float
    adx_threshold: float
    bb_width_threshold: float
    is_tradeable: bool


class RegimeClassifier:
    """Classifies market regime using ADX and Bollinger Band width.

    A TRENDING regime requires BOTH:
    - ADX >= adx_threshold (directional movement strength)
    - BB width >= bb_width_threshold (sufficient volatility expansion)

    Any other combination is classified as RANGING (not tradeable).
    """

    def __init__(
        self,
        adx_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        adx_threshold: float = 25.0,
        bb_width_threshold: float = 0.02,
    ) -> None:
        """Initialize the regime classifier.

        Args:
            adx_period: Period for ADX calculation (default 14).
            bb_period: Period for Bollinger Bands (default 20).
            bb_std: Standard deviation multiplier for Bollinger Bands (default 2.0).
            adx_threshold: Minimum ADX for trending regime (default 25.0).
            bb_width_threshold: Minimum BB width ratio for trending regime (default 0.02).
        """
        self.adx_period = adx_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.adx_threshold = adx_threshold
        self.bb_width_threshold = bb_width_threshold

    def classify(self, ohlcv_data: list[list]) -> RegimeResult:
        """Classify the current market regime from OHLCV data.

        Args:
            ohlcv_data: List of [timestamp, open, high, low, close, volume] candles.

        Returns:
            RegimeResult with classification and indicator values.
        """
        df = self._build_dataframe(ohlcv_data)
        df = self._compute_adx(df)
        df = self._compute_bb_width(df)

        latest = df.iloc[-1]
        adx = float(latest["adx"])
        bb_width = float(latest["bb_width"])

        is_trending = adx >= self.adx_threshold and bb_width >= self.bb_width_threshold
        regime = MarketRegime.TRENDING if is_trending else MarketRegime.RANGING

        logger.debug(
            "Regime: %s | ADX=%.2f (thresh=%.1f) | BB_width=%.4f (thresh=%.4f)",
            regime.value, adx, self.adx_threshold, bb_width, self.bb_width_threshold,
        )

        return RegimeResult(
            regime=regime,
            adx=adx,
            bb_width=bb_width,
            adx_threshold=self.adx_threshold,
            bb_width_threshold=self.bb_width_threshold,
            is_tradeable=is_trending,
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

    def _compute_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the Average Directional Index (ADX).

        ADX measures trend strength regardless of direction.
        Uses Wilder's smoothing (EMA with alpha=1/period).

        Args:
            df: DataFrame with 'high', 'low', 'close' columns.

        Returns:
            DataFrame with added 'adx' column.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm.abs()) & (minus_dm > 0), 0.0)

        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        alpha = 1.0 / self.adx_period
        atr_smooth = true_range.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_smooth.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_smooth.replace(0, np.nan))

        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        df["adx"] = dx.ewm(alpha=alpha, adjust=False).mean().fillna(0.0)
        return df

    def _compute_bb_width(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute normalized Bollinger Band width.

        BB width = (upper - lower) / middle, giving a relative measure
        of volatility that is comparable across different price levels.

        Args:
            df: DataFrame with a 'close' column.

        Returns:
            DataFrame with added 'bb_width' column.
        """
        rolling = df["close"].rolling(window=self.bb_period)
        middle = rolling.mean()
        std = rolling.std(ddof=1)

        upper = middle + (self.bb_std * std)
        lower = middle - (self.bb_std * std)

        df["bb_width"] = ((upper - lower) / middle.replace(0, np.nan)).fillna(0.0)
        return df
