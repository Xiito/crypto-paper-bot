"""Market hours scheduler for multi-asset trading.

Handles the different trading schedules across asset classes:
- Crypto: 24/7
- US Stocks/ETFs: Mon-Fri 9:30-16:00 ET (with pre/post market)

Provides waiting logic so the bot pauses during closed markets
rather than wasting API calls.
"""

import asyncio
import logging
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from bot.broker_base import AssetClass, MarketStatus

logger = logging.getLogger(__name__)

# US Eastern timezone for NYSE/NASDAQ hours
ET = ZoneInfo("America/New_York")

# Standard US equity market hours
US_MARKET_OPEN = dt_time(9, 30)
US_MARKET_CLOSE = dt_time(16, 0)
US_PRE_MARKET_OPEN = dt_time(4, 0)
US_AFTER_HOURS_CLOSE = dt_time(20, 0)

# Weekday constants
MON, TUE, WED, THU, FRI, SAT, SUN = range(7)
TRADING_DAYS = {MON, TUE, WED, THU, FRI}

# US market holidays (static list — update annually)
# These are the standard NYSE holidays for 2025-2026
US_HOLIDAYS_2025_2026 = {
    # 2025
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents' Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
    # 2026
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents' Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
}


class MarketHoursManager:
    """Manages market hours awareness for all asset classes.

    For crypto: always returns open.
    For stocks/ETFs: checks US market hours with holiday awareness.
    """

    def is_market_open(self, asset_class: AssetClass) -> MarketStatus:
        """Check if the market is currently open for the given asset class.

        Args:
            asset_class: The asset class to check.

        Returns:
            MarketStatus enum value.
        """
        if asset_class == AssetClass.CRYPTO:
            return MarketStatus.OPEN

        return self._check_us_equity_hours()

    def seconds_until_open(self, asset_class: AssetClass) -> float:
        """Calculate seconds until the market next opens.

        Args:
            asset_class: The asset class to check.

        Returns:
            Seconds until market opens (0 if already open).
        """
        if asset_class == AssetClass.CRYPTO:
            return 0.0

        return self._seconds_until_us_open()

    def get_next_open_time(self, asset_class: AssetClass) -> Optional[datetime]:
        """Get the next market open time in UTC.

        Args:
            asset_class: The asset class to check.

        Returns:
            Datetime of next market open, or None if always open.
        """
        if asset_class == AssetClass.CRYPTO:
            return None

        return self._next_us_open()

    async def wait_for_market_open(self, asset_class: AssetClass) -> None:
        """Block until the market is open.

        For crypto, returns immediately. For stocks, sleeps until
        the next trading session opens.

        Args:
            asset_class: The asset class to wait for.
        """
        if asset_class == AssetClass.CRYPTO:
            return

        wait_seconds = self.seconds_until_open(asset_class)
        if wait_seconds <= 0:
            return

        next_open = self.get_next_open_time(asset_class)
        logger.info(
            "Market closed for %s. Waiting %.0f minutes until %s",
            asset_class.value,
            wait_seconds / 60,
            next_open.strftime("%Y-%m-%d %H:%M ET") if next_open else "?",
        )

        # Sleep in chunks so we can log progress
        while wait_seconds > 0:
            chunk = min(wait_seconds, 300)  # Sleep max 5 min at a time
            await asyncio.sleep(chunk)
            wait_seconds -= chunk
            if wait_seconds > 60:
                logger.debug(
                    "Market opens in ~%.0f minutes", wait_seconds / 60
                )

    def _check_us_equity_hours(self) -> MarketStatus:
        """Check current US equity market status."""
        now_et = datetime.now(ET)

        # Weekend check
        if now_et.weekday() not in TRADING_DAYS:
            return MarketStatus.CLOSED

        # Holiday check
        date_str = now_et.strftime("%Y-%m-%d")
        if date_str in US_HOLIDAYS_2025_2026:
            return MarketStatus.CLOSED

        current_time = now_et.time()

        if US_MARKET_OPEN <= current_time < US_MARKET_CLOSE:
            return MarketStatus.OPEN
        elif US_PRE_MARKET_OPEN <= current_time < US_MARKET_OPEN:
            return MarketStatus.PRE_MARKET
        elif US_MARKET_CLOSE <= current_time < US_AFTER_HOURS_CLOSE:
            return MarketStatus.AFTER_HOURS
        else:
            return MarketStatus.CLOSED

    def _seconds_until_us_open(self) -> float:
        """Calculate seconds until next US market open."""
        now_et = datetime.now(ET)
        current_time = now_et.time()

        # If market is currently open, return 0
        if (
            now_et.weekday() in TRADING_DAYS
            and now_et.strftime("%Y-%m-%d") not in US_HOLIDAYS_2025_2026
            and US_MARKET_OPEN <= current_time < US_MARKET_CLOSE
        ):
            return 0.0

        next_open = self._next_us_open()
        if next_open is None:
            return 0.0

        delta = next_open - datetime.now(timezone.utc)
        return max(0.0, delta.total_seconds())

    def _next_us_open(self) -> Optional[datetime]:
        """Find the next US market open time in UTC."""
        now_et = datetime.now(ET)

        # If today is a trading day and market hasn't opened yet
        if (
            now_et.weekday() in TRADING_DAYS
            and now_et.strftime("%Y-%m-%d") not in US_HOLIDAYS_2025_2026
            and now_et.time() < US_MARKET_OPEN
        ):
            next_open_et = now_et.replace(
                hour=9, minute=30, second=0, microsecond=0
            )
            return next_open_et.astimezone(timezone.utc)

        # Otherwise, find the next trading day
        candidate = now_et + timedelta(days=1)
        for _ in range(10):  # Look up to 10 days ahead
            if (
                candidate.weekday() in TRADING_DAYS
                and candidate.strftime("%Y-%m-%d") not in US_HOLIDAYS_2025_2026
            ):
                next_open_et = candidate.replace(
                    hour=9, minute=30, second=0, microsecond=0
                )
                return next_open_et.astimezone(timezone.utc)
            candidate += timedelta(days=1)

        return None
