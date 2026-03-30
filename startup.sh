#!/bin/bash
# startup.sh — Azure Linux App Service startup script for MediQuery
#
# gunicorn manages process lifecycle; uvicorn worker handles async ASGI requests.
# Azure sets PORT; defaults to 8000 for local runs.

set -e

PORT="${PORT:-8000}"
WORKERS="${WEB_CONCURRENCY:-2}"

echo "Starting MediQuery on port $PORT with $WORKERS workers..."

exec gunicorn app.main:app \
    --bind "0.0.0.0:${PORT}" \
    --workers "${WORKERS}" \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info