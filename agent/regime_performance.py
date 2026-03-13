"""Regime performance aggregation module.

Queries the database for session-level statistics grouped by market
regime and updates the regime_performance table with rolling averages.
Generates the regime performance matrix used by the reflection agent
for parameter adjustment recommendations.
"""

import logging
from typing import Optional

from db.db_client import DatabaseClient

logger = logging.getLogger(__name__)


class RegimePerformanceAggregator:
    """Aggregates and maintains regime-level performance statistics.

    Queries all completed sessions grouped by their classified market
    regime and computes rolling win rates and average returns. Updates
    the regime_performance table in PostgreSQL for use by Grafana
    dashboards and the AI reflection agent.
    """

    def __init__(self, db: DatabaseClient) -> None:
        """Initialize the regime performance aggregator.

        Args:
            db: Database client for reading sessions and updating regime stats.
        """
        self._db = db

    async def update_regime_stats(self) -> list[dict]:
        """Recompute and update regime performance statistics from session data.

        Queries all sessions grouped by regime_label, computes per-regime
        metrics, and upserts the results into the regime_performance table.

        Returns:
            List of updated regime performance records.
        """
        stats = await self._db.get_regime_session_stats()

        if not stats:
            logger.info("No session data available for regime performance update")
            return []

        updated = []
        for row in stats:
            regime_label = row["regime_label"]
            total_sessions = row["session_count"]
            avg_return = float(row["avg_return"]) if row["avg_return"] is not None else 0.0
            win_rate = float(row["win_rate"]) if row["win_rate"] is not None else 0.0

            await self._db.upsert_regime_performance(
                regime_label=regime_label,
                total_sessions=total_sessions,
                avg_return_pct=avg_return,
                win_rate=win_rate,
            )

            updated.append({
                "regime_label": regime_label,
                "total_sessions": total_sessions,
                "avg_return_pct": avg_return,
                "win_rate": win_rate,
            })

        logger.info("Updated regime performance for %d regimes", len(updated))
        return updated

    async def get_performance_matrix(self) -> list[dict]:
        """Get the current regime performance matrix.

        Returns:
            List of regime performance records sorted by avg_return descending.
        """
        records = await self._db.get_all_regime_performance()
        logger.info("Retrieved regime performance matrix with %d entries", len(records))
        return records

    async def get_best_regime(self) -> Optional[dict]:
        """Get the best-performing regime by average return.

        Returns:
            The regime record with the highest avg_return_pct, or None.
        """
        records = await self._db.get_all_regime_performance()
        if not records:
            return None
        best = max(records, key=lambda r: float(r.get("avg_return_pct", 0)))
        return dict(best)

    async def get_worst_regime(self) -> Optional[dict]:
        """Get the worst-performing regime by average return.

        Returns:
            The regime record with the lowest avg_return_pct, or None.
        """
        records = await self._db.get_all_regime_performance()
        if not records:
            return None
        worst = min(records, key=lambda r: float(r.get("avg_return_pct", 0)))
        return dict(worst)

    async def should_recommend_parameter_change(self) -> bool:
        """Determine if enough data exists to recommend parameter changes.

        Requires at least 10 completed sessions and at least 2 distinct
        regimes observed to have enough statistical basis.

        Returns:
            True if parameter recommendations should be generated.
        """
        session_count = await self._db.get_session_count()
        if session_count < 10:
            logger.debug("Only %d sessions completed, need at least 10 for recommendations", session_count)
            return False

        records = await self._db.get_all_regime_performance()
        if len(records) < 2:
            logger.debug("Only %d regimes observed, need at least 2 for recommendations", len(records))
            return False

        return True
