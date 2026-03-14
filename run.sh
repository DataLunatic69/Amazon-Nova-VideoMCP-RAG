#!/usr/bin/env bash
# run.sh — start the full Video RAG backend (MCP server + API server)
#
# Usage:
#   ./run.sh           start both servers (default)
#   ./run.sh api       start the API server only
#   ./run.sh mcp       start the MCP tool server only
#   ./run.sh stop      kill both background servers started by this script
#
# Both servers read .env from the project root automatically.
# Logs are written to logs/mcp.log and logs/api.log.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"
APP_DIR="$REPO_ROOT/app"
LOG_DIR="$REPO_ROOT/logs"
PID_FILE="$REPO_ROOT/.backend.pids"

# ── Colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[run.sh]${RESET} $*"; }
success() { echo -e "${GREEN}[run.sh]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[run.sh]${RESET} $*"; }
error()   { echo -e "${RED}[run.sh]${RESET} $*" >&2; }

# ── Pre-flight checks ──────────────────────────────────────────────────────────
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  error "Virtual environment not found at $VENV_ACTIVATE"
  error "Create it with:  python -m venv .venv && .venv/bin/pip install -e ."
  exit 1
fi

if [[ ! -f "$REPO_ROOT/.env" ]]; then
  warn ".env not found — copying from .env.example"
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
  warn "Edit .env and add your AWS credentials before the servers will start correctly."
fi

# Change to the project root so any relative paths in sub-processes
# (e.g. Pixeltable DB paths, legacy code not using absolute paths) still work.
cd "$REPO_ROOT"

# Activate the virtual environment.
# shellcheck source=/dev/null
source "$VENV_ACTIVATE"

mkdir -p "$LOG_DIR"

# ── Helper: read a setting from .env (with fallback default) ──────────────────
_env_val() {
  local key="$1" default="${2:-}"
  local val
  val=$(grep -E "^${key}=" "$REPO_ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\"'" | xargs) || true
  echo "${val:-$default}"
}

MCP_PORT=$(_env_val MCP_PORT 9090)
API_PORT=$(_env_val API_PORT 8080)

# ── Stop command ───────────────────────────────────────────────────────────────
stop_servers() {
  if [[ ! -f "$PID_FILE" ]]; then
    warn "No PID file found — nothing to stop."
    return
  fi
  info "Stopping servers…"
  while IFS= read -r pid; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && info "  Stopped PID $pid"
    fi
  done < "$PID_FILE"
  rm -f "$PID_FILE"
  success "All servers stopped."
}

# ── Start MCP server ───────────────────────────────────────────────────────────
start_mcp() {
  info "Starting MCP tool server on port ${BOLD}${MCP_PORT}${RESET}…"
  info "  Log → $LOG_DIR/mcp.log"
  PYTHONPATH="$APP_DIR" \
    python -m video_rag.mcp.server \
      --host 0.0.0.0 \
      --port "$MCP_PORT" \
      --transport streamable-http \
    >> "$LOG_DIR/mcp.log" 2>&1 &
  echo $! >> "$PID_FILE"
  success "MCP server PID $! started."
}

# ── Start API server ───────────────────────────────────────────────────────────
start_api() {
  # Wait briefly for MCP to be reachable before launching the API.
  local max_wait=15 waited=0
  while ! curl -sf "http://localhost:${MCP_PORT}/mcp" >/dev/null 2>&1; do
    if (( waited >= max_wait )); then
      warn "MCP server did not respond in ${max_wait}s — starting API anyway."
      break
    fi
    sleep 1; (( waited++ )) || true
  done

  info "Starting FastAPI server on port ${BOLD}${API_PORT}${RESET}…"
  info "  Log → $LOG_DIR/api.log"
  PYTHONPATH="$APP_DIR" \
    uvicorn api.app:app \
      --host 0.0.0.0 \
      --port "$API_PORT" \
      --log-level info \
    >> "$LOG_DIR/api.log" 2>&1 &
  echo $! >> "$PID_FILE"
  success "API server PID $! started."
}

# ── Print status banner ────────────────────────────────────────────────────────
print_banner() {
  echo ""
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "${BOLD}  Video RAG — Amazon Nova Hackathon${RESET}"
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "  ${GREEN}MCP server ${RESET}→  http://localhost:${MCP_PORT}/mcp"
  echo -e "  ${GREEN}API server ${RESET}→  http://localhost:${API_PORT}"
  echo -e "  ${GREEN}API docs   ${RESET}→  http://localhost:${API_PORT}/docs"
  echo -e "  ${GREEN}Frontend   ${RESET}→  http://localhost:3000  (run: cd frontend && npm run dev)"
  echo -e "  ${GREEN}Logs       ${RESET}→  $LOG_DIR/"
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "  Stop with:  ${YELLOW}./run.sh stop${RESET}   or   ${YELLOW}Ctrl-C${RESET}"
  echo ""
}

# ── Trap Ctrl-C to clean up ────────────────────────────────────────────────────
trap 'echo ""; warn "Interrupted — stopping servers…"; stop_servers; exit 0' INT TERM

# ── Main dispatch ──────────────────────────────────────────────────────────────
MODE="${1:-both}"
rm -f "$PID_FILE"   # fresh PID list on each start

case "$MODE" in
  stop)
    stop_servers
    ;;
  mcp)
    start_mcp
    print_banner
    info "Tailing MCP log (Ctrl-C to stop)…"
    tail -f "$LOG_DIR/mcp.log"
    ;;
  api)
    start_api
    print_banner
    info "Tailing API log (Ctrl-C to stop)…"
    tail -f "$LOG_DIR/api.log"
    ;;
  both|"")
    start_mcp
    start_api
    print_banner
    info "Tailing logs (Ctrl-C to stop both servers)…"
    tail -f "$LOG_DIR/mcp.log" "$LOG_DIR/api.log"
    ;;
  *)
    error "Unknown command: '$MODE'"
    echo "Usage: $0 [both|api|mcp|stop]"
    exit 1
    ;;
esac
