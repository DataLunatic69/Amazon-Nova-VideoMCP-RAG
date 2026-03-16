#!/bin/bash
set -e

echo "Starting MCP server..."
cd /app/app
PYTHONPATH=/app/app python -m video_rag.mcp.server \
    --host 0.0.0.0 \
    --port 9090 \
    --transport streamable-http \
    >> /app/logs/mcp.log 2>&1 &

MCP_PID=$!
echo "MCP server started with PID $MCP_PID"

# Wait for MCP to be ready
echo "Waiting for MCP server to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:9090/mcp > /dev/null 2>&1; then
        echo "MCP server is ready."
        break
    fi
    sleep 2
done

echo "Starting API server..."
cd /app/app
PYTHONPATH=/app/app MCP_SERVER_URL=http://localhost:9090/mcp \
    gunicorn -c /app/gunicorn_config.py api.app:app \
    --bind 0.0.0.0:8080 \
    >> /app/logs/api.log 2>&1 &

API_PID=$!
echo "API server started with PID $API_PID"

# Tail logs so the container stays alive and HF can see output
tail -f /app/logs/mcp.log /app/logs/api.log