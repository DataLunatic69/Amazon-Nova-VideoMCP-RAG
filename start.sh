#!/bin/bash
set -e

echo "===== Application Startup ====="

# Create log directory
mkdir -p /app/logs

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
    echo "  attempt $i/30..."
    sleep 2
done

echo "Starting API server..."
cd /app/app
PYTHONPATH=/app/app MCP_SERVER_URL=http://localhost:9090/mcp \
    python -m gunicorn -c /app/gunicorn_config.py api.app:app \
    --bind 0.0.0.0:8080 \
    >> /app/logs/api.log 2>&1 &

API_PID=$!
echo "API server started with PID $API_PID"

echo "Both servers running."
echo "  MCP → http://localhost:9090/mcp"
echo "  API → http://localhost:8080"
echo "  Docs → http://localhost:8080/docs"

# Keep container alive and stream logs
tail -f /app/logs/mcp.log /app/logs/api.log