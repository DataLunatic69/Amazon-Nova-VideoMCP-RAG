"""
Gunicorn configuration for production deployment of the Video RAG API.

Run from the ``app/`` directory so all package imports resolve:

    cd app && gunicorn -c ../gunicorn_config.py api.app:app

Requirements:

    pip install gunicorn uvicorn[standard]
"""

import multiprocessing

# ── Worker ──────────────────────────────────────────────────────────────────────
# UvicornWorker handles ASGI (async FastAPI + WebSocket support).
worker_class = "uvicorn.workers.UvicornWorker"

# Standard heuristic for I/O-bound services: 2 × CPU + 1.
# Override via the WEB_CONCURRENCY environment variable.
workers = multiprocessing.cpu_count() * 2 + 1

# ── Binding ─────────────────────────────────────────────────────────────────────
bind = "0.0.0.0:8080"

# ── Timeouts ────────────────────────────────────────────────────────────────────
# Video ingestion (process_video) can take several minutes.
timeout = 300           # seconds before a hung worker is killed
keepalive = 5           # seconds to hold an idle keep-alive connection
graceful_timeout = 30   # seconds for workers to finish in-flight requests on SIGTERM

# ── Logging ─────────────────────────────────────────────────────────────────────
accesslog = "-"         # stdout
errorlog = "-"          # stderr
loglevel = "info"

# ── Process name ────────────────────────────────────────────────────────────────
proc_name = "video-rag-api"

# ── Reload ──────────────────────────────────────────────────────────────────────
# Disabled for production. Set to True only in local development.
reload = False
