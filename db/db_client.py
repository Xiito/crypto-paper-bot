"""Async PostgreSQL client using asyncpg with connection pooling.

Provides CRUD operations for all trading bot tables: sessions, trades,
reflections, regime_performance, and parameter_history.
"""

import json
import logging
import pathlib
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = pathlib.Path(__file__).parent / "migrations"


class DatabaseClient:
    """Async PostgreSQL client with connection pooling and auto-migration."""

    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10) -> None:
        """Initialize the database client.

        Args:
            dsn: PostgreSQL connection DSN string.
            min_size: Minimum number of connections in the pool.
            max_size: Maximum number of connections in the pool.
        """
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Create the connection pool and run migrations."""
        logger.info("Connecting to PostgreSQL at %s", self._dsn.split("@")[-1])
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
        )
        await self._run_migrations()
        logger.info("Database connection pool established (min=%d, max=%d)", self._min_size, self._max_size)

    async def disconnect(self) -> None:
        """Close all connections in the pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Database connection pool closed")

    async def health_check(self) -> bool:
        """Check if the database connection is alive.

        Returns:
            True if the connection is healthy, False otherwise.
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as exc:
            logger.error("Database health check failed: %s", exc)
            return False

    async def _run_migrations(self) -> None:
        """Run all SQL migration files in order."""
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        if not migration_files:
            logger.warning("No migration files found in %s", MIGRATIONS_DIR)
            return

        async with self._pool.acquire() as conn:
            for migration_file in migration_files:
                logger.info("Running migration: %s", migration_file.name)
                sql = migration_file.read_text()
                await conn.execute(sql)
                logger.info("Migration %s applied successfully", migration_file.name)

    # ---- Session Operations ----

    async def create_session(self, session_date: date, starting_capital: float) -> UUID:
        """Create a new trading session.

        Args:
            session_date: The date of the trading session.
            starting_capital: The starting paper capital amount.

        Returns:
            The UUID of the newly created session.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO sessions (date, starting_capital)
                VALUES ($1, $2)
                ON CONFLICT (date) DO UPDATE SET starting_capital = $2
                RETURNING id
                """,
                session_date,
                Decimal(str(starting_capital)),
            )
            session_id = row["id"]
            logger.info("Created session %s for date %s with capital $%.2f", session_id, session_date, starting_capital)
            return session_id

    async def close_session(
        self,
        session_id: UUID,
        ending_capital: float,
        total_trades: int,
        win_count: int,
        loss_count: int,
        session_return_pct: float,
        regime_label: Optional[str] = None,
    ) -> None:
        """Close a trading session with final statistics.

        Args:
            session_id: The session UUID to close.
            ending_capital: Final capital at session end.
            total_trades: Total number of trades executed.
            win_count: Number of winning trades.
            loss_count: Number of losing trades.
            session_return_pct: Session return as a percentage.
            regime_label: Dominant market regime during the session.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE sessions SET
                    ending_capital = $2,
                    total_trades = $3,
                    win_count = $4,
                    loss_count = $5,
                    session_return_pct = $6,
                    regime_label = $7
                WHERE id = $1
                """,
                session_id,
                Decimal(str(ending_capital)),
                total_trades,
                win_count,
                loss_count,
                Decimal(str(session_return_pct)),
                regime_label,
            )
            logger.info(
                "Closed session %s: capital=$%.2f, trades=%d, win_rate=%.1f%%",
                session_id,
                ending_capital,
                total_trades,
                (win_count / total_trades * 100) if total_trades > 0 else 0,
            )

    async def get_session_by_date(self, session_date: date) -> Optional[dict]:
        """Get a session record by date.

        Args:
            session_date: The date to look up.

        Returns:
            Session record as a dict, or None if not found.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM sessions WHERE date = $1", session_date)
            return dict(row) if row else None

    async def get_session_count(self) -> int:
        """Get the total number of completed sessions.

        Returns:
            The count of sessions with a non-null ending_capital.
        """
        async with self._pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM sessions WHERE ending_capital IS NOT NULL")

    # ---- Trade Operations ----

    async def insert_trade(
        self,
        session_id: UUID,
        pair: str,
        direction: str,
        entry_price: float,
        quantity: float,
        signal_ema_cross: bool,
        signal_rsi: float,
        signal_atr: float,
        regime_tag: Optional[str] = None,
    ) -> UUID:
        """Insert a new trade record (entry only, no exit yet).

        Args:
            session_id: The session this trade belongs to.
            pair: Trading pair (e.g., 'BTC/USDT').
            direction: Trade direction ('LONG' or 'SHORT').
            entry_price: Entry price of the trade.
            quantity: Position size.
            signal_ema_cross: Whether EMA crossover signal was present.
            signal_rsi: RSI value at entry.
            signal_atr: ATR value at entry.
            regime_tag: Market regime at time of entry.

        Returns:
            The UUID of the newly created trade.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO trades (
                    session_id, pair, direction, entry_price, quantity,
                    signal_ema_cross, signal_rsi, signal_atr, regime_tag
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
                """,
                session_id,
                pair,
                direction,
                Decimal(str(entry_price)),
                Decimal(str(quantity)),
                signal_ema_cross,
                Decimal(str(signal_rsi)),
                Decimal(str(signal_atr)),
                regime_tag,
            )
            trade_id = row["id"]
            logger.info("Inserted trade %s: %s %s @ %.8f", trade_id, direction, pair, entry_price)
            return trade_id

    async def close_trade(
        self,
        trade_id: UUID,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        hold_duration_seconds: int,
        exit_reason: str,
    ) -> None:
        """Close an open trade with exit details.

        Args:
            trade_id: The trade UUID to close.
            exit_price: Exit price of the trade.
            pnl: Absolute profit/loss in USDT.
            pnl_pct: Profit/loss as a percentage.
            hold_duration_seconds: How long the position was held.
            exit_reason: Why the trade was closed.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE trades SET
                    exit_price = $2,
                    pnl = $3,
                    pnl_pct = $4,
                    hold_duration_seconds = $5,
                    exit_reason = $6
                WHERE id = $1
                """,
                trade_id,
                Decimal(str(exit_price)),
                Decimal(str(pnl)),
                Decimal(str(pnl_pct)),
                hold_duration_seconds,
                exit_reason,
            )
            logger.info("Closed trade %s: exit=%.8f, pnl=%.4f (%.2f%%), reason=%s", trade_id, exit_price, pnl, pnl_pct, exit_reason)

    async def get_open_trades(self, session_id: UUID) -> list[dict]:
        """Get all open (unclosed) trades for a session.

        Args:
            session_id: The session to query.

        Returns:
            List of open trade records as dicts.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM trades WHERE session_id = $1 AND exit_price IS NULL",
                session_id,
            )
            return [dict(r) for r in rows]

    async def get_session_trades(self, session_id: UUID) -> list[dict]:
        """Get all trades for a session (open and closed).

        Args:
            session_id: The session to query.

        Returns:
            List of trade records as dicts.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM trades WHERE session_id = $1 ORDER BY created_at",
                session_id,
            )
            return [dict(r) for r in rows]

    # ---- Reflection Operations ----

    async def insert_reflection(
        self,
        session_id: UUID,
        regime_label: str,
        regime_confidence: float,
        losses_analysis: dict,
        wins_analysis: dict,
        parameter_suggestion: Optional[dict] = None,
    ) -> UUID:
        """Insert an LLM reflection report for a session.

        Args:
            session_id: The session this reflection belongs to.
            regime_label: Classified market regime.
            regime_confidence: Confidence score for the regime classification.
            losses_analysis: JSON analysis of losing trades.
            wins_analysis: JSON analysis of winning trades.
            parameter_suggestion: Optional parameter adjustment suggestion.

        Returns:
            The UUID of the newly created reflection.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO reflections (
                    session_id, regime_label, regime_confidence,
                    losses_analysis, wins_analysis, parameter_suggestion
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                session_id,
                regime_label,
                Decimal(str(regime_confidence)),
                json.dumps(losses_analysis),
                json.dumps(wins_analysis),
                json.dumps(parameter_suggestion) if parameter_suggestion else None,
            )
            reflection_id = row["id"]
            logger.info("Inserted reflection %s for session %s (regime=%s, confidence=%.1f%%)", reflection_id, session_id, regime_label, regime_confidence)
            return reflection_id

    # ---- Regime Performance Operations ----

    async def upsert_regime_performance(
        self,
        regime_label: str,
        total_sessions: int,
        avg_return_pct: float,
        win_rate: float,
    ) -> None:
        """Insert or update regime performance statistics.

        Args:
            regime_label: The regime label to update.
            total_sessions: Total number of sessions with this regime.
            avg_return_pct: Average return percentage across sessions.
            win_rate: Win rate as a percentage.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO regime_performance (regime_label, total_sessions, avg_return_pct, win_rate, last_updated)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (regime_label)
                DO UPDATE SET
                    total_sessions = $2,
                    avg_return_pct = $3,
                    win_rate = $4,
                    last_updated = NOW()
                """,
                regime_label,
                total_sessions,
                Decimal(str(avg_return_pct)),
                Decimal(str(win_rate)),
            )
            logger.info("Upserted regime performance: %s (sessions=%d, avg_return=%.2f%%, win_rate=%.1f%%)", regime_label, total_sessions, avg_return_pct, win_rate)

    async def get_all_regime_performance(self) -> list[dict]:
        """Get all regime performance records.

        Returns:
            List of regime performance records as dicts.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM regime_performance ORDER BY avg_return_pct DESC")
            return [dict(r) for r in rows]

    # ---- Parameter History Operations ----

    async def insert_parameter_change(
        self,
        parameter_name: str,
        old_value: str,
        new_value: str,
        suggested_by_agent: bool = True,
    ) -> UUID:
        """Record a parameter change suggested by the AI agent.

        Args:
            parameter_name: Name of the parameter changed.
            old_value: Previous value.
            new_value: New suggested value.
            suggested_by_agent: Whether this was AI-suggested.

        Returns:
            The UUID of the parameter history record.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO parameter_history (parameter_name, old_value, new_value, suggested_by_agent)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                parameter_name,
                old_value,
                new_value,
                suggested_by_agent,
            )
            param_id = row["id"]
            logger.info("Recorded parameter change: %s (%s -> %s, agent=%s)", parameter_name, old_value, new_value, suggested_by_agent)
            return param_id

    async def get_regime_session_stats(self) -> list[dict]:
        """Get aggregated session statistics grouped by regime.

        Returns:
            List of dicts with regime_label, session_count, avg_return, win_rate.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    regime_label,
                    COUNT(*) AS session_count,
                    AVG(session_return_pct) AS avg_return,
                    AVG(CASE WHEN total_trades > 0
                        THEN (win_count::float / total_trades * 100)
                        ELSE 0 END) AS win_rate
                FROM sessions
                WHERE regime_label IS NOT NULL
                  AND ending_capital IS NOT NULL
                GROUP BY regime_label
                ORDER BY avg_return DESC
                """
            )
            return [dict(r) for r in rows]
