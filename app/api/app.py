"""
FastAPI application factory for the Video RAG API.

Usage
─────
Development (from the ``app/`` directory):

    uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload

Production (installed package entry point):

    serve-api

Or directly:

    python -m api.app
"""

import contextlib
import uuid
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.routers import api_router
from config import get_settings
from video_rag.agent.nova_agent import NovaAgent
from logging_config import setup_logging, get_logger, log_startup_banner, LoggingContext

# Initialize logging first
setup_logging()
logger = get_logger("API")

settings = get_settings()


# ── Request logging middleware ─────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests with timing and context."""
    
    async def dispatch(self, request: Request, call_next):
        import time
        
        # Skip WebSocket connections (handled separately)
        if request.url.path.startswith("/api/v1/voice/stream"):
            logger.info(f"WebSocket connection | path={request.url.path}")
            return await call_next(request)
        
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()
        
        # Skip logging for health checks to reduce noise
        if request.url.path == "/health":
            return await call_next(request)
        
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )
        
        try:
            response = await call_next(request)
            
            duration = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration,
            )
            
            # Add request ID to response headers for tracing
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as exc:
            duration = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(exc),
                duration_ms=duration,
            )
            raise


# ── Lifespan ───────────────────────────────────────────────────────────────────

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan — initialise and tear down shared process-level state.

    Everything attached to ``app.state`` here is accessible from every request
    handler via ``request.app.state.*``.
    """
    log_startup_banner(
        app_name="Video RAG API",
        version="0.1.0",
        endpoints={
            "API": f"http://{settings.API_HOST}:{settings.API_PORT}",
            "Docs": f"http://{settings.API_HOST}:{settings.API_PORT}/docs",
            "MCP": settings.MCP_SERVER_URL,
        },
    )

    # ── NovaAgent (one per process) ────────────────────────────────────────────
    # We create the agent here but deliberately do NOT call agent.setup() yet.
    # setup() opens a connection to the MCP server; if MCP isn't up yet the
    # API process would fail to start.  Instead, setup() is called lazily on
    # the first request (each endpoint handler calls `await agent.setup()`
    # which is idempotent).
    agent = NovaAgent(mcp_server_url=settings.MCP_SERVER_URL)
    app.state.agent = agent
    logger.info(
        "Agent created",
        session_id=agent._session_id,
        mcp_server_url=settings.MCP_SERVER_URL,
    )

    # ── Background task state dictionary ──────────────────────────────────────
    # Maps task_id → TaskStatus for video ingestion jobs kicked off via
    # POST /video/process.  Lives in memory; resets on restart (by design —
    # completed ingestion is persisted in the Pixeltable registry).
    app.state.task_states = {}

    logger.info(
        "Video RAG API ready",
        api_host=settings.API_HOST,
        api_port=settings.API_PORT,
        aws_region=settings.AWS_REGION,
    )
    yield  # ── serve requests ──────────────────────────────────────────────────

    logger.info("Shutting down Video RAG API …")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Build, configure, and return the FastAPI application."""
    application = FastAPI(
        title="Video RAG API",
        description=(
            "Multimodal video retrieval-augmented generation powered by Amazon Nova.\n\n"
            "**Features**\n"
            "- Upload and ingest videos (frame extraction, audio transcription, "
            "scene captioning, multimodal embedding via Amazon Titan).\n"
            "- Retrieve clips by natural-language query or reference image.\n"
            "- Ask factual questions about video content (RAG with Nova Pro).\n"

            "**Quick start**\n"
            "1. `POST /api/v1/video/upload` — upload a video file.\n"
            "2. `POST /api/v1/video/process` — ingest it (runs in background).\n"
            "3. `GET  /api/v1/video/task-status/{task_id}` — wait for completion.\n"
            "4. `POST /api/v1/chat` — ask questions or find clips.\n"
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ───────────────────────────────────────────────────────────────────
    # CORS must be added FIRST (before other middleware) to ensure WebSocket
    # upgrade requests are properly handled with correct CORS headers.
    # Allow all origins for the hackathon demo.
    # Tighten ``allow_origins`` to specific domains before going to production.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Transcript",
            "X-Message",
            "X-Audio-Path",
            "X-Clip-Path",
        ],
    )

    # ── Request logging middleware ─────────────────────────────────────────────
    # Added AFTER CORS to avoid interfering with WebSocket upgrades
    application.add_middleware(RequestLoggingMiddleware)

    # ── Routers ────────────────────────────────────────────────────────────────
    application.include_router(api_router)

    # ── Health check ───────────────────────────────────────────────────────────
    @application.get("/health", tags=["ops"], summary="Health / liveness check")
    async def health():
        """Returns ``{"status": "ok"}`` when the API process is alive."""
        return JSONResponse({"status": "ok"})

    return application


# Module-level app instance (used by uvicorn / gunicorn).
app = create_app()


# ── CLI entry point ────────────────────────────────────────────────────────────

def run_api() -> None:
    """
    Start the API server.

    Called by the ``serve-api`` console script defined in pyproject.toml.
    """
    uvicorn.run(
        "api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run_api()
