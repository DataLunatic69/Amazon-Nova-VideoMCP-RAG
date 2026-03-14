"""
Repository root entry point.

Adds ``app/`` to sys.path so all package imports resolve correctly when
the script is run from the repository root rather than from inside ``app/``.

Usage
─────
Run the full stack (two separate terminals):

    # Terminal 1 — MCP tool server (port 9090)
    python main.py mcp

    # Terminal 2 — FastAPI API server (port 8080)
    python main.py api          # default when no subcommand is given
    python main.py              # same as above

Or use the package console scripts (after ``pip install -e .``):

    serve-mcp   # MCP server
    serve-api   # API server
"""

import sys
from pathlib import Path

# ── Ensure app/ is on sys.path ─────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import click  # noqa: E402


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Video RAG — Amazon Nova hackathon project."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(api)


@cli.command()
def api() -> None:
    """Start the FastAPI API server (default, port 8080)."""
    from api.app import run_api

    run_api()


@cli.command()
def mcp() -> None:
    """Start the FastMCP tool server (port 9090)."""
    from video_rag.mcp.server import run_mcp

    run_mcp(standalone_mode=False)


if __name__ == "__main__":
    cli()
