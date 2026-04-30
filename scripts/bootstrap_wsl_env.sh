#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "[INFO] This script is intended for WSL2 Ubuntu, but continuing anyway."
fi

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "[INFO] Installing Python 3.12 tooling..."
  sudo apt update
  sudo apt install -y software-properties-common curl ca-certificates
  sudo add-apt-repository -y ppa:deadsnakes/ppa || true
  sudo apt update
  sudo apt install -y python3.12 python3.12-venv python3.12-dev build-essential
fi

python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . psutil asyncpg "psycopg[binary]"

if [ ! -f .env ]; then
  cp .env.example .env
fi

# Align .env with docker-compose.yml. ClickHouse default user has an empty password here.
python - <<'PY'
from pathlib import Path
p = Path('.env')
text = p.read_text() if p.exists() else ''
updates = {
    'CH_HOST': 'localhost',
    'CH_PORT': '8123',
    'CH_DATABASE': 'market_data',
    'CH_USER': 'default',
    'CH_PASSWORD': '',
    'PG_HOST': 'localhost',
    'PG_PORT': '5432',
    'PG_DATABASE': 'geoxiao',
    'PG_USER': 'geoxiao',
    'PG_PASSWORD': 'secret',
    'PG_DSN': 'postgresql+asyncpg://geoxiao:secret@localhost:5432/geoxiao',
    'OPTUNA_STORAGE': 'postgresql://geoxiao:secret@localhost:5432/geoxiao',
}
lines = text.splitlines()
seen = set()
out = []
for line in lines:
    if '=' in line and not line.lstrip().startswith('#'):
        key = line.split('=', 1)[0].strip()
        if key in updates:
            out.append(f'{key}={updates[key]}')
            seen.add(key)
        else:
            out.append(line)
    else:
        out.append(line)
for key, value in updates.items():
    if key not in seen:
        out.append(f'{key}={value}')
p.write_text('\n'.join(out) + '\n')
PY

docker compose up -d

echo "[INFO] Waiting for database health checks..."
for i in {1..60}; do
  pg_status=$(docker inspect --format='{{.State.Health.Status}}' geoxiao-postgres 2>/dev/null || echo starting)
  ch_status=$(docker inspect --format='{{.State.Health.Status}}' geoxiao-clickhouse 2>/dev/null || echo starting)
  echo "[INFO] postgres=$pg_status clickhouse=$ch_status"
  if [ "$pg_status" = "healthy" ] && [ "$ch_status" = "healthy" ]; then
    break
  fi
  sleep 2
done

python check_env.py
