# ---- Stage 1: Builder ----
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local

COPY config.py .
COPY bot/ bot/
COPY agent/ agent/
COPY db/ db/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import asyncio; asyncio.run(__import__('db.db_client', fromlist=['DatabaseClient']).DatabaseClient.__new__(__import__('db.db_client', fromlist=['DatabaseClient']).DatabaseClient).health_check())" || exit 1

CMD ["python", "-m", "bot.main"]
