#!/usr/bin/env bash
# Run the typoon-media container locally with R2 mocked as a filesystem.
# Useful for fast-iteration debugging of the prepare stage without CF
# deploy round-trips.
#
#   bash scripts/r2-pull.sh raw/<job>/source.zip   # one-time
#   bash scripts/local-media.sh                    # starts uvicorn :8766
#   curl -X POST 'http://127.0.0.1:8766/prepare?job_id=<job>&zip_key=raw/<job>/source.zip&strategy=auto'

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export R2_MOUNT="${R2_MOUNT:-/tmp/r2-mock}"
mkdir -p "$R2_MOUNT"
PORT="${PORT:-8766}"
cd "$ROOT/workers/media/container"
exec "$ROOT/.venv/bin/python" -m uvicorn main:app --host 127.0.0.1 --port "$PORT" --log-level info
