"""
Comprehensive logging configuration for Video RAG.

Features
────────
- Structured JSON logging for production (LOG_FORMAT=json)
- Pretty colored console logging for development (LOG_FORMAT=pretty, default)
- File rotation with retention policy
- Request/Response logging middleware support
- Performance timing decorators
- Contextual logging (request_id, session_id, video_path)

Usage
─────
    from logging_config import setup_logging, get_logger
    
    # Call once at startup
    setup_logging()
    
    # Get a contextualized logger
    logger = get_logger("MyModule")
    logger.info("Processing started", extra={"video_path": "/path/to/video.mp4"})
"""

import os
import sys
import time
import functools
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Optional
from contextvars import ContextVar

from loguru import logger

# ── Context variables for request tracking ─────────────────────────────────────
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
session_id_ctx: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
video_path_ctx: ContextVar[Optional[str]] = ContextVar("video_path", default=None)


# ── Configuration ──────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_FORMAT = os.getenv("LOG_FORMAT", "pretty")  # "pretty" or "json"
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_ROTATION = os.getenv("LOG_ROTATION", "10 MB")
LOG_RETENTION = os.getenv("LOG_RETENTION", "7 days")


# ── Format strings ─────────────────────────────────────────────────────────────

PRETTY_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[module]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
    "{extra[context]}"
)

JSON_FORMAT = (
    '{{"timestamp": "{time:YYYY-MM-DDTHH:mm:ss.SSSZ}", '
    '"level": "{level}", '
    '"module": "{extra[module]}", '
    '"function": "{function}", '
    '"line": {line}, '
    '"message": "{message}", '
    '"request_id": "{extra[request_id]}", '
    '"session_id": "{extra[session_id]}", '
    '"video_path": "{extra[video_path]}", '
    '"extra": {extra[json_extra]}}}'
)


def _format_context(record: dict) -> str:
    """Format extra context for pretty printing."""
    extra = record.get("extra", {})
    parts = []
    
    if extra.get("request_id"):
        parts.append(f"req={extra['request_id']}")
    if extra.get("session_id"):
        parts.append(f"sess={extra['session_id']}")
    if extra.get("video_path"):
        video = Path(extra["video_path"]).name
        parts.append(f"video={video}")
    if extra.get("duration_ms"):
        parts.append(f"⏱{extra['duration_ms']:.1f}ms")
    if extra.get("tool_name"):
        parts.append(f"tool={extra['tool_name']}")
    if "use_tool" in extra:
        parts.append(f"use_tool={'✅' if extra['use_tool'] else '❌'}")
    if "used_tool" in extra:
        parts.append(f"used_tool={'✅' if extra['used_tool'] else '❌'}")
    if extra.get("model_id"):
        model = extra["model_id"].split("/")[-1].split(":")[0]
        parts.append(f"model={model}")
    
    if parts:
        return " | " + " ".join(parts)
    return ""


def _patcher(record: dict) -> dict:
    """Patch log records with context variables and formatting."""
    record["extra"]["module"] = record["extra"].get("name", record.get("name", "unknown"))
    record["extra"]["request_id"] = record["extra"].get("request_id") or request_id_ctx.get()
    record["extra"]["session_id"] = record["extra"].get("session_id") or session_id_ctx.get()
    record["extra"]["video_path"] = record["extra"].get("video_path") or video_path_ctx.get()
    record["extra"]["context"] = _format_context(record)
    
    # For JSON format, serialize extra fields
    json_extra = {
        k: v for k, v in record["extra"].items()
        if k not in ("module", "request_id", "session_id", "video_path", "context", "json_extra", "name")
        and v is not None
    }
    record["extra"]["json_extra"] = str(json_extra).replace("'", '"') if json_extra else "{}"
    
    return record


def setup_logging() -> None:
    """
    Configure loguru with comprehensive logging setup.
    
    Call this once at application startup (in main.py or app factory).
    """
    # Remove default handler
    logger.remove()
    
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Add patcher for context
    logger.configure(patcher=_patcher)
    
    # ── Console handler ────────────────────────────────────────────────────────
    fmt = PRETTY_FORMAT if LOG_FORMAT == "pretty" else JSON_FORMAT
    logger.add(
        sys.stderr,
        format=fmt,
        level=LOG_LEVEL,
        colorize=(LOG_FORMAT == "pretty"),
        backtrace=True,
        diagnose=True,
    )
    
    # ── File handlers ──────────────────────────────────────────────────────────
    # All logs
    logger.add(
        LOG_DIR / "app.log",
        format=JSON_FORMAT,
        level="DEBUG",
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        compression="gz",
        serialize=False,
    )
    
    # Error logs only
    logger.add(
        LOG_DIR / "error.log",
        format=JSON_FORMAT,
        level="ERROR",
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        compression="gz",
        serialize=False,
    )
    
    # Performance logs
    logger.add(
        LOG_DIR / "performance.log",
        format=JSON_FORMAT,
        level="INFO",
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        compression="gz",
        filter=lambda r: r["extra"].get("performance"),
    )
    
    logger.info(
        "Logging initialized",
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        log_dir=str(LOG_DIR),
    )


def get_logger(name: str):
    """
    Get a contextualized logger for a module.
    
    Args:
        name: Module or component name (e.g., "NovaAgent", "VideoProcessor")
    
    Returns:
        A bound logger instance with the module name attached.
    """
    return logger.bind(name=name)


# ── Performance timing decorator ───────────────────────────────────────────────

def log_performance(
    operation: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    threshold_ms: float = 0,
) -> Callable:
    """
    Decorator to log function execution time and optionally args/results.
    
    Args:
        operation: Custom operation name (defaults to function name)
        log_args: Whether to log function arguments
        log_result: Whether to log return value (truncated)
        threshold_ms: Only log if execution exceeds this threshold
    
    Example:
        @log_performance(operation="embed_image", threshold_ms=100)
        def embed_image(image: Image) -> np.ndarray:
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__
        func_logger = get_logger(func.__module__.split(".")[-1])
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                
                if duration_ms >= threshold_ms:
                    extra = {
                        "performance": True,
                        "operation": op_name,
                        "duration_ms": duration_ms,
                        "success": True,
                    }
                    if log_args:
                        extra["args"] = _truncate(str(args)[:200])
                        extra["kwargs"] = _truncate(str(kwargs)[:200])
                    if log_result:
                        extra["result"] = _truncate(str(result)[:200])
                    
                    func_logger.info(f"⏱ {op_name} completed", **extra)
                
                return result
                
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000
                func_logger.error(
                    f"⏱ {op_name} failed: {exc}",
                    performance=True,
                    operation=op_name,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(exc),
                )
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                
                if duration_ms >= threshold_ms:
                    extra = {
                        "performance": True,
                        "operation": op_name,
                        "duration_ms": duration_ms,
                        "success": True,
                    }
                    if log_args:
                        extra["args"] = _truncate(str(args)[:200])
                        extra["kwargs"] = _truncate(str(kwargs)[:200])
                    if log_result:
                        extra["result"] = _truncate(str(result)[:200])
                    
                    func_logger.info(f"⏱ {op_name} completed", **extra)
                
                return result
                
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000
                func_logger.error(
                    f"⏱ {op_name} failed: {exc}",
                    performance=True,
                    operation=op_name,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(exc),
                )
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def _truncate(s: str, max_len: int = 200) -> str:
    """Truncate a string for logging."""
    return s[:max_len] + "…" if len(s) > max_len else s


# ── Request logging context manager ────────────────────────────────────────────

class LoggingContext:
    """
    Context manager for request-scoped logging.
    
    Example:
        async with LoggingContext(request_id="abc123", video_path="/path/to/video.mp4"):
            logger.info("Processing request")  # Automatically includes context
    """
    
    def __init__(
        self,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        video_path: Optional[str] = None,
    ):
        self.request_id = request_id
        self.session_id = session_id
        self.video_path = video_path
        self._tokens = []
    
    def __enter__(self):
        if self.request_id:
            self._tokens.append(request_id_ctx.set(self.request_id))
        if self.session_id:
            self._tokens.append(session_id_ctx.set(self.session_id))
        if self.video_path:
            self._tokens.append(video_path_ctx.set(self.video_path))
        return self
    
    def __exit__(self, *args):
        for token in reversed(self._tokens):
            # Reset context vars (loguru doesn't need explicit reset)
            pass
    
    async def __aenter__(self):
        return self.__enter__()
    
    async def __aexit__(self, *args):
        return self.__exit__(*args)


# ── Startup banner ─────────────────────────────────────────────────────────────

def log_startup_banner(
    app_name: str,
    version: str = "0.1.0",
    endpoints: Optional[dict] = None,
) -> None:
    """Log a formatted startup banner."""
    banner = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {app_name} v{version}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Environment: {os.getenv('ENVIRONMENT', 'development')}
  Log Level:   {LOG_LEVEL}
  Log Format:  {LOG_FORMAT}
"""
    if endpoints:
        for name, url in endpoints.items():
            banner += f"  {name}: {url}\n"
    
    banner += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    logger.info(banner)
