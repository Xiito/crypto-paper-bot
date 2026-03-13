# AI-Powered Crypto Paper Trading Bot

An AI-powered crypto paper trading bot that runs daily sessions on live Binance Testnet market data. Each day starts with $1,000 paper capital. Every trade is logged to PostgreSQL, and an LLM reflection agent analyzes performance at end-of-day.

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Binance     │    │   Bot Core   │    │  PostgreSQL  │
│  Testnet API │◄──►│  (Strategy,  │◄──►│  (Sessions,  │
│              │    │  Risk Mgmt,  │    │  Trades,     │
│              │    │  Execution)  │    │  Reflections)│
└──────────────┘    └──────┬───────┘    └──────┬───────┘
                           │                   │
                    ┌──────▼───────┐    ┌──────▼───────┐
                    │  AI Agent    │    │   Grafana    │
                    │  (GPT-4o /   │    │  Dashboard   │
                    │   Ollama)    │    │              │
                    └──────────────┘    └──────────────┘
```

## Features

- **Paper Trading**: Trades BTC/USDT and ETH/USDT on Binance Testnet (no real money)
- **Signal Engine**: EMA(9/21) crossover + RSI(14) range + ATR(14) volatility gate
- **Risk Management**: 2% max risk per trade, 3 max concurrent positions, -10% daily hard stop
- **AI Reflection**: GPT-4o analyzes every session — classifies regime, diagnoses losses, identifies winning patterns
- **Regime Detection**: Classifies markets as strong bull / bear trend / sideways range / high volatility spike
- **Performance Matrix**: Every 10 sessions, generates a regime performance matrix and parameter recommendation
- **Full Observability**: Grafana dashboard with daily returns, win rate, regime heatmap, and AI insights

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Binance Testnet API keys ([get them here](https://testnet.binance.vision/))
- OpenAI API key (or local Ollama installation)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/crypto-paper-bot.git
cd crypto-paper-bot

# 2. Create your environment file
cp .env.example .env
# Edit .env with your API keys

# 3. Start all services
docker compose up -d

# 4. Check logs
docker compose logs -f bot

# 5. Open Grafana dashboard
# http://localhost:3000 (admin/admin)
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run linting
ruff check .
ruff format --check .

# Run tests
pytest tests/ -v --asyncio-mode=auto

# Start with dev overrides (debug logging, source mounting)
docker compose up -d
```

## Project Structure

```
├── bot/
│   ├── main.py                 # Entry point, session lifecycle
│   ├── strategy.py             # EMA/RSI/ATR signal generation
│   ├── risk_manager.py         # Position sizing, loss caps
│   ├── order_executor.py       # Binance Testnet via CCXT
│   ├── regime_classifier.py    # Market regime detection
│   ├── session_manager.py      # Daily $1K reset logic
│   └── trade_logger.py         # Trade → PostgreSQL persistence
├── agent/
│   ├── reflection_agent.py     # End-of-day LLM analysis
│   ├── regime_performance.py   # Regime matrix aggregation
│   └── prompts.py              # LLM prompt templates
├── db/
│   ├── schema.sql              # Full PostgreSQL schema
│   ├── db_client.py            # Async PostgreSQL client
│   └── migrations/
│       └── 001_initial.sql     # Initial migration
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── postgres.yml    # Grafana → PostgreSQL config
│       └── dashboards/
│           ├── dashboard.yml   # Dashboard provisioning config
│           └── trading_bot.json # Full dashboard definition
├── tests/
│   ├── test_strategy.py        # Signal generation tests
│   ├── test_risk_manager.py    # Risk management tests
│   └── test_reflection_agent.py # LLM agent tests (mocked)
├── .github/workflows/
│   └── ci.yml                  # CI/CD: lint → test → build → push
├── Dockerfile                  # Multi-stage Python image
├── docker-compose.yml          # All services
├── docker-compose.override.yml # Dev overrides
├── config.py                   # Centralized env-based config
├── requirements.txt            # Pinned dependencies
├── .env.example                # Environment template
└── .gitignore                  # Excludes .env, __pycache__, etc.
```

## Trading Strategy

| Condition       | Long Signal                          | Short Signal                         |
|----------------|--------------------------------------|--------------------------------------|
| EMA Crossover  | EMA(9) crosses above EMA(21)         | EMA(9) crosses below EMA(21)         |
| RSI Range      | RSI(14) between 45–65                | RSI(14) between 35–55                |
| ATR Volatility | ATR(14) above 25th percentile        | ATR(14) above 25th percentile        |

### Risk Rules

- Max 2% capital at risk per trade
- Max 3 concurrent open positions
- Daily hard stop at -10% session loss
- Stop-loss: 1.5× ATR from entry price

## AI Reflection Agent

At end of each session, the agent:
1. Receives all trades as structured JSON
2. Classifies the market regime (EMA slope + ATR percentile + BB width)
3. Analyzes each losing trade (wrong regime, weak signal, poor timing, stop too tight, overextension)
4. Identifies what conditions aligned for winning trades
5. Stores structured JSON reflection in PostgreSQL

Every 10 sessions:
- Generates a regime performance matrix (win rate + avg return per regime)
- Outputs one concrete parameter adjustment recommendation

### LLM Backend

Configure via `.env`:
- `LLM_PROVIDER=openai` — Uses GPT-4o with structured JSON output
- `LLM_PROVIDER=ollama` — Uses local Ollama (e.g., llama3)

## CI/CD Pipeline

```
Push → Lint (ruff) → Test (pytest + PostgreSQL) → Build Docker → Push to GHCR
```

- Runs on every push to `main` and `develop`
- Docker image only pushed on `main` merges
- Uses GitHub Container Registry (GHCR)

## Grafana Dashboard

Access at `http://localhost:3000` (default: admin/admin)

Panels:
- **Overview**: Current P&L, win rate, total sessions, active positions
- **Performance**: Daily return curve, cumulative P&L, win rate over time
- **Regime Analysis**: Distribution pie chart, performance matrix table, return by regime
- **Trade Analysis**: P&L histogram, exit reason breakdown, hold duration by pair
- **AI Insights**: Latest reflection reports, parameter change history

## Environment Variables

See [`.env.example`](.env.example) for the full list with descriptions.

## License

MIT
