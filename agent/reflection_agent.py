"""End-of-day LLM reflection agent.

Triggered automatically after each trading session to analyze trades,
classify the market regime, and produce structured JSON reflection
reports. Supports both OpenAI GPT-4o and local Ollama as LLM backends.
"""

import asyncio
import json
import logging
from typing import Any, Optional
from uuid import UUID

import aiohttp

from agent.prompts import (
    PARAMETER_RECOMMENDATION_SYSTEM,
    PARAMETER_RECOMMENDATION_USER,
    SESSION_REFLECTION_SYSTEM,
    SESSION_REFLECTION_USER,
)
from agent.regime_performance import RegimePerformanceAggregator
from db.db_client import DatabaseClient

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_RETRY_DELAY = 2.0


class ReflectionAgent:
    """AI reflection agent that analyzes completed trading sessions.

    After each session, this agent:
    1. Loads all trades from the session
    2. Sends the trade log to the LLM for structured analysis
    3. Stores the reflection report in PostgreSQL
    4. Every 10 sessions, generates a parameter adjustment recommendation

    Supports OpenAI GPT-4o (primary) and Ollama (fallback) as LLM backends.
    """

    def __init__(
        self,
        db: DatabaseClient,
        provider: str = "openai",
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3",
        max_retries: int = 3,
        timeout_seconds: int = 60,
        trading_config: Optional[dict] = None,
    ) -> None:
        """Initialize the reflection agent.

        Args:
            db: Database client for reading trades and storing reflections.
            provider: LLM provider ('openai' or 'ollama').
            openai_api_key: OpenAI API key (required if provider is 'openai').
            openai_model: OpenAI model name.
            ollama_base_url: Ollama API base URL.
            ollama_model: Ollama model name.
            max_retries: Maximum LLM API retry attempts.
            timeout_seconds: HTTP request timeout.
            trading_config: Current trading parameters for recommendation context.
        """
        self._db = db
        self._provider = provider
        self._openai_api_key = openai_api_key
        self._openai_model = openai_model
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._ollama_model = ollama_model
        self._max_retries = max_retries
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._trading_config = trading_config or {}
        self._regime_aggregator = RegimePerformanceAggregator(db)

    async def run_reflection(
        self,
        session_id: UUID,
        session_date: str,
        starting_capital: float,
        ending_capital: float,
        session_return_pct: float,
        total_trades: int,
        win_count: int,
        loss_count: int,
    ) -> Optional[dict]:
        """Run the end-of-day reflection analysis for a completed session.

        Args:
            session_id: The completed session's UUID.
            session_date: Session date as string.
            starting_capital: Starting capital.
            ending_capital: Ending capital.
            session_return_pct: Session return percentage.
            total_trades: Total trades executed.
            win_count: Winning trade count.
            loss_count: Losing trade count.

        Returns:
            The reflection report as a dict, or None if analysis failed.
        """
        logger.info("Starting reflection analysis for session %s (%s)", session_id, session_date)

        trades = await self._db.get_session_trades(session_id)
        if not trades:
            logger.warning("No trades found for session %s, skipping reflection", session_id)
            return None

        trades_json = self._serialize_trades(trades)

        prompt = SESSION_REFLECTION_USER.format(
            session_date=session_date,
            starting_capital=starting_capital,
            ending_capital=ending_capital,
            session_return_pct=session_return_pct,
            total_trades=total_trades,
            win_count=win_count,
            loss_count=loss_count,
            trades_json=trades_json,
        )

        reflection = await self._call_llm(SESSION_REFLECTION_SYSTEM, prompt)
        if reflection is None:
            logger.error("LLM returned no response for session %s reflection", session_id)
            return None

        parsed = self._parse_json_response(reflection)
        if parsed is None:
            logger.error("Failed to parse LLM reflection response for session %s", session_id)
            return None

        regime_label = parsed.get("regime_label", "unknown")
        regime_confidence = float(parsed.get("regime_confidence", 0))
        losses_analysis = parsed.get("losses_analysis", [])
        wins_analysis = parsed.get("wins_analysis", [])

        await self._db.insert_reflection(
            session_id=session_id,
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            losses_analysis={"trades": losses_analysis},
            wins_analysis={"trades": wins_analysis},
            parameter_suggestion=parsed.get("parameter_suggestion"),
        )

        await self._regime_aggregator.update_regime_stats()

        session_count = await self._db.get_session_count()
        if session_count > 0 and session_count % 10 == 0:
            logger.info("Session count %d is a multiple of 10, generating parameter recommendation", session_count)
            await self._generate_parameter_recommendation(session_count)

        logger.info(
            "Reflection complete for session %s: regime=%s (%.1f%% confidence)",
            session_id, regime_label, regime_confidence,
        )
        return parsed

    async def _generate_parameter_recommendation(self, session_count: int) -> Optional[dict]:
        """Generate a parameter adjustment recommendation every 10 sessions.

        Args:
            session_count: Total number of completed sessions.

        Returns:
            The parameter recommendation as a dict, or None if generation failed.
        """
        regime_stats = await self._db.get_regime_session_stats()
        if not regime_stats:
            logger.warning("No regime stats available for parameter recommendation")
            return None

        regime_matrix_json = json.dumps(
            [
                {
                    "regime": row["regime_label"],
                    "sessions": row["session_count"],
                    "avg_return": float(row["avg_return"]) if row["avg_return"] else 0,
                    "win_rate": float(row["win_rate"]) if row["win_rate"] else 0,
                }
                for row in regime_stats
            ],
            indent=2,
        )

        recent_returns_json = json.dumps([], indent=2)

        config = self._trading_config
        prompt = PARAMETER_RECOMMENDATION_USER.format(
            session_count=session_count,
            regime_matrix_json=regime_matrix_json,
            ema_fast=config.get("ema_fast_period", 9),
            ema_slow=config.get("ema_slow_period", 21),
            rsi_period=config.get("rsi_period", 14),
            rsi_long_min=config.get("rsi_long_min", 45),
            rsi_long_max=config.get("rsi_long_max", 65),
            rsi_short_min=config.get("rsi_short_min", 35),
            rsi_short_max=config.get("rsi_short_max", 55),
            atr_period=config.get("atr_period", 14),
            max_risk_pct=config.get("max_risk_per_trade_pct", 0.02),
            recent_returns_json=recent_returns_json,
        )

        response = await self._call_llm(PARAMETER_RECOMMENDATION_SYSTEM, prompt)
        if response is None:
            return None

        parsed = self._parse_json_response(response)
        if parsed is None:
            return None

        param_name = parsed.get("parameter_name", "unknown")
        old_value = parsed.get("current_value", "")
        new_value = parsed.get("recommended_value", "")

        if param_name and old_value and new_value:
            await self._db.insert_parameter_change(
                parameter_name=param_name,
                old_value=old_value,
                new_value=new_value,
                suggested_by_agent=True,
            )
            logger.info(
                "Parameter recommendation stored: %s (%s -> %s), confidence=%.1f%%",
                param_name, old_value, new_value, parsed.get("confidence", 0),
            )

        return parsed

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call the configured LLM backend with retry logic.

        Args:
            system_prompt: System-level instructions for the LLM.
            user_prompt: User-level prompt with session data.

        Returns:
            Raw LLM response text, or None if all attempts failed.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                if self._provider == "openai":
                    return await self._call_openai(system_prompt, user_prompt)
                elif self._provider == "ollama":
                    return await self._call_ollama(system_prompt, user_prompt)
                else:
                    logger.error("Unknown LLM provider: %s", self._provider)
                    return None
            except asyncio.TimeoutError:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning("LLM timeout (attempt %d/%d), retrying in %.1fs", attempt, self._max_retries, delay)
                await asyncio.sleep(delay)
            except aiohttp.ClientError as exc:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning("LLM HTTP error (attempt %d/%d): %s", attempt, self._max_retries, exc)
                await asyncio.sleep(delay)
            except Exception as exc:
                logger.error("Unexpected LLM error (attempt %d/%d): %s", attempt, self._max_retries, exc)
                if attempt == self._max_retries:
                    return None
                await asyncio.sleep(BASE_RETRY_DELAY)

        return None

    async def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call the OpenAI API.

        Args:
            system_prompt: System message.
            user_prompt: User message.

        Returns:
            The assistant's response content.

        Raises:
            aiohttp.ClientError: On HTTP errors.
            RuntimeError: On non-200 responses.
        """
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._openai_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                },
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"OpenAI API returned {resp.status}: {body}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    async def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Ollama API.

        Args:
            system_prompt: System message.
            user_prompt: User message.

        Returns:
            The model's response content.

        Raises:
            aiohttp.ClientError: On HTTP errors.
            RuntimeError: On non-200 responses.
        """
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(
                f"{self._ollama_base_url}/api/chat",
                json={
                    "model": self._ollama_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "format": "json",
                },
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Ollama API returned {resp.status}: {body}")
                data = await resp.json()
                return data["message"]["content"]

    def _serialize_trades(self, trades: list[dict]) -> str:
        """Serialize trade records to a JSON string for LLM input.

        Converts UUID, Decimal, and datetime objects to JSON-safe types.

        Args:
            trades: List of trade record dicts from the database.

        Returns:
            Pretty-printed JSON string of the trade list.
        """
        serializable = []
        for trade in trades:
            record = {}
            for key, value in trade.items():
                if isinstance(value, UUID):
                    record[key] = str(value)
                elif hasattr(value, "as_integer_ratio"):
                    record[key] = float(value)
                elif hasattr(value, "isoformat"):
                    record[key] = value.isoformat()
                else:
                    record[key] = value
            serializable.append(record)
        return json.dumps(serializable, indent=2)

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Parse a JSON response from the LLM, handling common formatting issues.

        Args:
            response: Raw LLM response string.

        Returns:
            Parsed JSON dict, or None if parsing failed.
        """
        response = response.strip()

        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON response: %s\nResponse was: %s", exc, response[:500])
            return None
