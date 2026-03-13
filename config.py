"""Centralized configuration loaded from environment variables.

All settings are validated at startup. Missing required variables
raise clear error messages. Defaults are provided for optional settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _require_env(key: str) -> str:
    """Return an environment variable or raise with a helpful message."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in all required values."
        )
    return value


def _env(key: str, default: str = "") -> str:
    """Return an environment variable with a fallback default."""
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    """Return an environment variable as an integer."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise EnvironmentError(f"Environment variable '{key}' must be an integer, got '{raw}'.")


def _env_float(key: str, default: float) -> float:
    """Return an environment variable as a float."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise EnvironmentError(f"Environment variable '{key}' must be a float, got '{raw}'.")


def _env_bool(key: str, default: bool) -> bool:
    """Return an environment variable as a boolean."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.lower() in ("true", "1", "yes")


@dataclass(frozen=True)
class BinanceConfig:
    """Binance Testnet API configuration."""

    api_key: str
    api_secret: str
    testnet: bool = True
    rate_limit: bool = True


@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL database configuration."""

    host: str
    port: int
    name: str
    user: str
    password: str
    min_pool_size: int = 2
    max_pool_size: int = 10

    @property
    def dsn(self) -> str:
        """Return a PostgreSQL connection DSN string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass(frozen=True)
class LLMConfig:
    """LLM configuration for the reflection agent."""

    provider: str  # "openai" or "ollama"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    max_retries: int = 3
    timeout_seconds: int = 60


@dataclass(frozen=True)
class TradingConfig:
    """Trading strategy and risk management parameters."""

    pairs: list = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    starting_capital: float = 1000.0
    max_risk_per_trade_pct: float = 0.02
    max_concurrent_positions: int = 3
    daily_hard_stop_pct: float = -0.10
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    rsi_period: int = 14
    atr_period: int = 14
    rsi_long_min: float = 45.0
    rsi_long_max: float = 65.0
    rsi_short_min: float = 35.0
    rsi_short_max: float = 55.0
    ohlcv_timeframe: str = "1m"
    ohlcv_limit: int = 100
    signal_check_interval_seconds: int = 60
    session_start_hour_utc: int = 0
    session_end_hour_utc: int = 23
    session_end_minute_utc: int = 55


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    json_format: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration combining all sub-configs."""

    binance: BinanceConfig
    database: DatabaseConfig
    llm: LLMConfig
    trading: TradingConfig
    logging: LoggingConfig


def load_config() -> AppConfig:
    """Load and validate all configuration from environment variables.

    Returns:
        AppConfig: Fully validated application configuration.

    Raises:
        EnvironmentError: If required environment variables are missing or invalid.
    """
    binance = BinanceConfig(
        api_key=_require_env("BINANCE_API_KEY"),
        api_secret=_require_env("BINANCE_API_SECRET"),
        testnet=_env_bool("BINANCE_TESTNET", True),
        rate_limit=_env_bool("BINANCE_RATE_LIMIT", True),
    )

    database = DatabaseConfig(
        host=_env("POSTGRES_HOST", "localhost"),
        port=_env_int("POSTGRES_PORT", 5432),
        name=_env("POSTGRES_DB", "trading_bot"),
        user=_env("POSTGRES_USER", "postgres"),
        password=_env("POSTGRES_PASSWORD", "postgres"),
        min_pool_size=_env_int("DB_MIN_POOL_SIZE", 2),
        max_pool_size=_env_int("DB_MAX_POOL_SIZE", 10),
    )

    llm_provider = _env("LLM_PROVIDER", "openai")
    llm = LLMConfig(
        provider=llm_provider,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=_env("OPENAI_MODEL", "gpt-4o"),
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=_env("OLLAMA_MODEL", "llama3"),
        max_retries=_env_int("LLM_MAX_RETRIES", 3),
        timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 60),
    )

    if llm_provider == "openai" and not llm.openai_api_key:
        raise EnvironmentError(
            "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set. "
            "Set OPENAI_API_KEY or change LLM_PROVIDER to 'ollama'."
        )

    trading_pairs_raw = _env("TRADING_PAIRS", "BTC/USDT,ETH/USDT")
    trading_pairs = [p.strip() for p in trading_pairs_raw.split(",") if p.strip()]

    trading = TradingConfig(
        pairs=trading_pairs,
        starting_capital=_env_float("STARTING_CAPITAL", 1000.0),
        max_risk_per_trade_pct=_env_float("MAX_RISK_PER_TRADE_PCT", 0.02),
        max_concurrent_positions=_env_int("MAX_CONCURRENT_POSITIONS", 3),
        daily_hard_stop_pct=_env_float("DAILY_HARD_STOP_PCT", -0.10),
        ema_fast_period=_env_int("EMA_FAST_PERIOD", 9),
        ema_slow_period=_env_int("EMA_SLOW_PERIOD", 21),
        rsi_period=_env_int("RSI_PERIOD", 14),
        atr_period=_env_int("ATR_PERIOD", 14),
        rsi_long_min=_env_float("RSI_LONG_MIN", 45.0),
        rsi_long_max=_env_float("RSI_LONG_MAX", 65.0),
        rsi_short_min=_env_float("RSI_SHORT_MIN", 35.0),
        rsi_short_max=_env_float("RSI_SHORT_MAX", 55.0),
        ohlcv_timeframe=_env("OHLCV_TIMEFRAME", "1m"),
        ohlcv_limit=_env_int("OHLCV_LIMIT", 100),
        signal_check_interval_seconds=_env_int("SIGNAL_CHECK_INTERVAL_SECONDS", 60),
        session_start_hour_utc=_env_int("SESSION_START_HOUR_UTC", 0),
        session_end_hour_utc=_env_int("SESSION_END_HOUR_UTC", 23),
        session_end_minute_utc=_env_int("SESSION_END_MINUTE_UTC", 55),
    )

    logging_cfg = LoggingConfig(
        level=_env("LOG_LEVEL", "INFO"),
        json_format=_env_bool("LOG_JSON_FORMAT", True),
    )

    return AppConfig(
        binance=binance,
        database=database,
        llm=llm,
        trading=trading,
        logging=logging_cfg,
    )
