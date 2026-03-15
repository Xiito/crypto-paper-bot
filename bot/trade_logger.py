"""Trade logging module for structured JSON trade records.

Logs trade entries, exits, and session summaries to a JSONL file
for post-session analysis and performance review.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from bot.order_executor import OrderResult
from bot.risk_manager import PositionSizeResult
from bot.strategy import TradeSignal

logger = logging.getLogger(__name__)


class TradeLogger:
    """Logs structured trade records to a JSONL file.

    Each record is a JSON object on its own line, making the log
    easy to parse for analysis or display in a dashboard.
    """

    def __init__(self, log_path: str = "trades.jsonl") -> None:
        """Initialize the trade logger.

        Args:
            log_path: Path to the output JSONL file.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_start = time.time()
        logger.info("TradeLogger initialized: %s", self.log_path)

    def log_entry(
        self,
        trade_id: str,
        signal: TradeSignal,
        size_result: PositionSizeResult,
        order: OrderResult,
    ) -> None:
        """Log a trade entry event.

        Args:
            trade_id: Unique trade identifier.
            signal: The trade signal that triggered the entry.
            size_result: The position sizing result.
            order: The order execution result.
        """
        record = {
            "event": "entry",
            "timestamp": time.time(),
            "trade_id": trade_id,
            "pair": signal.pair,
            "direction": signal.direction.value,
            "entry_price": order.price,
            "quantity": order.quantity,
            "stop_loss_price": signal.stop_loss_price,
            "risk_amount": size_result.risk_amount,
            "signal_strength": signal.strength,
            "ema_fast": signal.ema_fast,
            "ema_slow": signal.ema_slow,
            "rsi": signal.rsi,
            "atr": signal.atr,
            "order_id": order.order_id,
        }
        self._write(record)

    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        reason: str = "signal",
    ) -> None:
        """Log a trade exit event.

        Args:
            trade_id: Unique trade identifier.
            exit_price: The price at which the trade was closed.
            pnl: Realized profit and loss.
            reason: Reason for exit ('signal', 'stop_loss', 'manual').
        """
        record = {
            "event": "exit",
            "timestamp": time.time(),
            "trade_id": trade_id,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
        }
        self._write(record)

    def log_session_summary(self, stats: dict) -> None:
        """Log a session summary record.

        Args:
            stats: Session statistics dict from RiskManager.get_session_stats().
        """
        record = {
            "event": "session_summary",
            "timestamp": time.time(),
            "session_duration_seconds": time.time() - self._session_start,
            **stats,
        }
        self._write(record)
        logger.info("Session summary logged: return=%.2f%%, trades=%d", stats.get("session_return_pct", 0) * 100, stats.get("total_trades", 0))

    def _write(self, record: dict) -> None:
        """Write a record to the JSONL file.

        Args:
            record: Dict to serialize as a JSON line.
        """
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("Failed to write trade record: %s", exc)
