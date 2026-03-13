"""Unit tests for the AI reflection agent.

Tests cover LLM API calls (OpenAI and Ollama), retry logic,
JSON parsing, regime classification, and parameter recommendations.
All external HTTP calls are mocked.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from agent.reflection_agent import ReflectionAgent


SAMPLE_SESSION_ID = uuid4()
SAMPLE_REFLECTION_RESPONSE = json.dumps({
    "regime_label": "strong_bull",
    "regime_confidence": 82.5,
    "regime_reasoning": "EMA(9) crossed above EMA(21) with strong momentum.",
    "losses_analysis": [
        {
            "trade_id": str(uuid4()),
            "pair": "BTC/USDT",
            "pnl": -45.20,
            "cause": "poor_timing",
            "explanation": "Entry was made at RSI=68, near overbought territory.",
        }
    ],
    "wins_analysis": [
        {
            "trade_id": str(uuid4()),
            "pair": "ETH/USDT",
            "pnl": 120.50,
            "aligned_conditions": "EMA crossover confirmed with RSI=52, ideal range.",
        }
    ],
    "session_summary": "Strong bullish session with 3 wins and 1 loss.",
    "key_observation": "RSI entries near 65+ showed poor risk/reward.",
})

SAMPLE_PARAM_RESPONSE = json.dumps({
    "parameter_name": "rsi_long_max",
    "current_value": "65",
    "recommended_value": "62",
    "reasoning": "Strong bull sessions showed losses when RSI exceeded 63.",
    "expected_impact": "Reduce late entries in overbought conditions.",
    "confidence": 75.0,
})


def _make_mock_db(
    trades=None,
    session_count=5,
    regime_stats=None,
):
    """Create a mock DatabaseClient for testing.

    Args:
        trades: List of trade records to return.
        session_count: Number of sessions to report.
        regime_stats: Regime performance stats to return.

    Returns:
        MagicMock configured as a DatabaseClient.
    """
    db = MagicMock()
    db.get_session_trades = AsyncMock(return_value=trades or [
        {
            "trade_id": str(uuid4()),
            "pair": "BTC/USDT",
            "direction": "long",
            "entry_price": 50000.0,
            "exit_price": 51000.0,
            "pnl": 100.0,
            "opened_at": "2024-01-15T09:00:00",
            "closed_at": "2024-01-15T10:00:00",
        }
    ])
    db.insert_reflection = AsyncMock(return_value=None)
    db.get_session_count = AsyncMock(return_value=session_count)
    db.get_regime_session_stats = AsyncMock(return_value=regime_stats or [
        {"regime_label": "strong_bull", "session_count": 5, "avg_return": 0.025, "win_rate": 0.7},
        {"regime_label": "sideways_range", "session_count": 3, "avg_return": -0.005, "win_rate": 0.4},
    ])
    db.upsert_regime_performance = AsyncMock(return_value=None)
    db.insert_parameter_change = AsyncMock(return_value=None)
    return db


def _make_agent(db, provider="openai", session_count=5):
    """Create a ReflectionAgent instance for testing.

    Args:
        db: Mock database client.
        provider: LLM provider ('openai' or 'ollama').
        session_count: Session count for mock db.

    Returns:
        ReflectionAgent configured for testing.
    """
    return ReflectionAgent(
        db=db,
        provider=provider,
        openai_api_key="test-key-123",
        openai_model="gpt-4o",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
        max_retries=3,
        timeout_seconds=30,
        trading_config={
            "ema_fast_period": 9,
            "ema_slow_period": 21,
            "rsi_period": 14,
            "rsi_long_min": 45,
            "rsi_long_max": 65,
            "rsi_short_min": 35,
            "rsi_short_max": 55,
            "atr_period": 14,
            "max_risk_per_trade_pct": 0.02,
        },
    )


class TestReflectionAgent:
    """Tests for the ReflectionAgent class."""

    @pytest.mark.asyncio
    async def test_run_reflection_success_openai(self):
        """Test successful reflection run with OpenAI backend."""
        db = _make_mock_db()
        agent = _make_agent(db)

        with patch.object(agent, "_call_openai", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = SAMPLE_REFLECTION_RESPONSE
            result = await agent.run_reflection(
                session_id=SAMPLE_SESSION_ID,
                session_date="2024-01-15",
                starting_capital=10000.0,
                ending_capital=10250.0,
                session_return_pct=2.5,
                total_trades=4,
                win_count=3,
                loss_count=1,
            )

        assert result is not None
        assert result["regime_label"] == "strong_bull"
        assert result["regime_confidence"] == 82.5
        db.insert_reflection.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_reflection_success_ollama(self):
        """Test successful reflection run with Ollama backend."""
        db = _make_mock_db()
        agent = _make_agent(db, provider="ollama")

        with patch.object(agent, "_call_ollama", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = SAMPLE_REFLECTION_RESPONSE
            result = await agent.run_reflection(
                session_id=SAMPLE_SESSION_ID,
                session_date="2024-01-15",
                starting_capital=10000.0,
                ending_capital=10250.0,
                session_return_pct=2.5,
                total_trades=4,
                win_count=3,
                loss_count=1,
            )

        assert result is not None
        assert result["regime_label"] == "strong_bull"

    @pytest.mark.asyncio
    async def test_run_reflection_no_trades_returns_none(self):
        """Test that reflection returns None when no trades exist."""
        db = _make_mock_db(trades=[])
        agent = _make_agent(db)
        result = await agent.run_reflection(
            session_id=SAMPLE_SESSION_ID,
            session_date="2024-01-15",
            starting_capital=10000.0,
            ending_capital=10000.0,
            session_return_pct=0.0,
            total_trades=0,
            win_count=0,
            loss_count=0,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_run_reflection_llm_failure_returns_none(self):
        """Test that reflection returns None when LLM fails all retries."""
        db = _make_mock_db()
        agent = _make_agent(db)

        with patch.object(agent, "_call_openai", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("LLM unavailable")
            result = await agent.run_reflection(
                session_id=SAMPLE_SESSION_ID,
                session_date="2024-01-15",
                starting_capital=10000.0,
                ending_capital=9800.0,
                session_return_pct=-2.0,
                total_trades=2,
                win_count=0,
                loss_count=2,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_parameter_recommendation_triggered_at_10_sessions(self):
        """Test that parameter recommendation is triggered every 10 sessions."""
        db = _make_mock_db(session_count=10)
        agent = _make_agent(db, session_count=10)

        with patch.object(agent, "_call_openai", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [SAMPLE_REFLECTION_RESPONSE, SAMPLE_PARAM_RESPONSE]
            await agent.run_reflection(
                session_id=SAMPLE_SESSION_ID,
                session_date="2024-01-15",
                starting_capital=10000.0,
                ending_capital=10250.0,
                session_return_pct=2.5,
                total_trades=4,
                win_count=3,
                loss_count=1,
            )

        db.insert_parameter_change.assert_called_once()

    @pytest.mark.asyncio
    async def test_parameter_recommendation_not_triggered_at_5_sessions(self):
        """Test that parameter recommendation is NOT triggered at non-multiple sessions."""
        db = _make_mock_db(session_count=5)
        agent = _make_agent(db)

        with patch.object(agent, "_call_openai", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = SAMPLE_REFLECTION_RESPONSE
            await agent.run_reflection(
                session_id=SAMPLE_SESSION_ID,
                session_date="2024-01-15",
                starting_capital=10000.0,
                ending_capital=10100.0,
                session_return_pct=1.0,
                total_trades=3,
                win_count=2,
                loss_count=1,
            )

        db.insert_parameter_change.assert_not_called()

    def test_parse_json_response_valid(self):
        """Test JSON parsing of valid response."""
        agent = _make_agent(_make_mock_db())
        result = agent._parse_json_response(SAMPLE_REFLECTION_RESPONSE)
        assert result is not None
        assert result["regime_label"] == "strong_bull"

    def test_parse_json_response_with_backticks(self):
        """Test JSON parsing strips markdown code fences."""
        agent = _make_agent(_make_mock_db())
        wrapped = f"```json\n{SAMPLE_REFLECTION_RESPONSE}\n```"
        result = agent._parse_json_response(wrapped)
        assert result is not None
        assert result["regime_label"] == "strong_bull"

    def test_parse_json_response_invalid_returns_none(self):
        """Test that invalid JSON returns None."""
        agent = _make_agent(_make_mock_db())
        result = agent._parse_json_response("this is not json")
        assert result is None

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test that LLM calls are retried on timeout."""
        db = _make_mock_db()
        agent = _make_agent(db)
        agent._max_retries = 3

        call_count = 0

        async def failing_then_success(system, user):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise asyncio.TimeoutError()
            return SAMPLE_REFLECTION_RESPONSE

        with patch.object(agent, "_call_openai", side_effect=failing_then_success):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await agent.run_reflection(
                    session_id=SAMPLE_SESSION_ID,
                    session_date="2024-01-15",
                    starting_capital=10000.0,
                    ending_capital=10100.0,
                    session_return_pct=1.0,
                    total_trades=2,
                    win_count=2,
                    loss_count=0,
                )

        assert result is not None
        assert call_count == 3

    def test_serialize_trades(self):
        """Test trade serialization handles UUID and datetime."""
        from datetime import datetime
        agent = _make_agent(_make_mock_db())
        trades = [
            {
                "trade_id": UUID("12345678-1234-5678-1234-567812345678"),
                "pair": "BTC/USDT",
                "pnl": 100.0,
                "opened_at": datetime(2024, 1, 15, 9, 0, 0),
            }
        ]
        result = agent._serialize_trades(trades)
        parsed = json.loads(result)
        assert parsed[0]["trade_id"] == "12345678-1234-5678-1234-567812345678"
        assert "2024-01-15" in parsed[0]["opened_at"]
