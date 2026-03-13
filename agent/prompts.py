"""LLM prompt templates for the AI reflection agent.

All prompt templates are stored as string constants to ensure
consistency and easy versioning. Templates use Python string
formatting placeholders.
"""

SESSION_REFLECTION_SYSTEM = """You are a quantitative trading analyst AI. Your job is to analyze \
a completed trading session and produce a structured reflection report.

You must output valid JSON matching the exact schema specified. Do not include any text outside the JSON object.

Analysis guidelines:
- Be specific about market conditions, not generic
- Reference actual price levels and indicator values from the trade data
- Classify each trade's outcome cause with high precision
- Provide actionable insights, not vague recommendations"""

SESSION_REFLECTION_USER = """Analyze this completed trading session and produce a structured JSON reflection report.

## Session Summary
- Date: {session_date}
- Starting Capital: ${starting_capital:.2f}
- Ending Capital: ${ending_capital:.2f}
- Session Return: {session_return_pct:.2f}%
- Total Trades: {total_trades}
- Wins: {win_count} | Losses: {loss_count}

## Trade Log (JSON)
{trades_json}

## Required Output Schema
Respond with ONLY a JSON object matching this structure:
{{
    "regime_label": "strong_bull | bear_trend | sideways_range | high_volatility_spike",
    "regime_confidence": <float 0-100>,
    "regime_reasoning": "<1-2 sentence explanation of why this regime was classified>",
    "losses_analysis": [
        {{
            "trade_id": "<uuid>",
            "pair": "<pair>",
            "pnl": <float>,
            "cause": "wrong_regime | weak_signal | poor_timing | stop_too_tight | overextension",
            "explanation": "<specific explanation referencing indicator values>"
        }}
    ],
    "wins_analysis": [
        {{
            "trade_id": "<uuid>",
            "pair": "<pair>",
            "pnl": <float>,
            "aligned_conditions": "<what market conditions made this trade work>"
        }}
    ],
    "session_summary": "<2-3 sentence overall assessment>",
    "key_observation": "<single most important insight from this session>"
}}"""

PARAMETER_RECOMMENDATION_SYSTEM = """You are a quantitative trading strategy optimizer. \
You analyze regime performance data across multiple trading sessions and recommend \
specific parameter adjustments to improve strategy performance.

You must output valid JSON. Be conservative with recommendations — only suggest changes \
when the data clearly supports them. One recommendation at a time."""

PARAMETER_RECOMMENDATION_USER = """Based on the following regime performance matrix from \
the last {session_count} trading sessions, recommend ONE specific parameter adjustment.

## Regime Performance Matrix
{regime_matrix_json}

## Current Strategy Parameters
- EMA Fast Period: {ema_fast}
- EMA Slow Period: {ema_slow}
- RSI Period: {rsi_period}
- RSI Long Range: {rsi_long_min}-{rsi_long_max}
- RSI Short Range: {rsi_short_min}-{rsi_short_max}
- ATR Period: {atr_period}
- Max Risk Per Trade: {max_risk_pct:.1%}

## Recent Session Returns
{recent_returns_json}

## Required Output Schema
Respond with ONLY a JSON object:
{{
    "parameter_name": "<exact parameter name, e.g. 'rsi_long_max'>",
    "current_value": "<current value as string>",
    "recommended_value": "<new value as string>",
    "reasoning": "<2-3 sentences explaining why, referencing specific regime data>",
    "expected_impact": "<1 sentence on expected improvement>",
    "confidence": <float 0-100>
}}"""

REGIME_CLASSIFICATION_SYSTEM = """You are a market regime classifier. Analyze the provided \
indicator data and classify the current market regime. Output valid JSON only."""

REGIME_CLASSIFICATION_USER = """Classify the market regime based on these indicators:

- EMA(21) Slope (normalized): {ema_slope:.6f}
- ATR(14) Percentile Rank: {atr_percentile:.2%}
- Bollinger Band Width: {bb_width:.6f}
- Recent Price Action: {price_action_summary}

Respond with ONLY a JSON object:
{{
    "regime": "strong_bull | bear_trend | sideways_range | high_volatility_spike",
    "confidence": <float 0-100>,
    "reasoning": "<brief explanation>"
}}"""
