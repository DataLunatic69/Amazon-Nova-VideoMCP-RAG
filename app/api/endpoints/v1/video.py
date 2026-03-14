"""
Video router — handles everything related to video files.

Endpoints
─────────
POST /upload-video          Upload a video file; returns its server-side path.
POST /process-video         Kick off background ingestion (Pixeltable pipeline).
GET  /task-status/{task_id} Poll the status of a background ingestion task.
GET  /media/{file_path}     Serve a generated clip or uploaded video.
"""

import asyncio
import shutil
import time
from enum import Enum
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from api.schemas import (
    ProcessVideoRequest,
    ProcessVideoResponse,
    TaskStatusResponse,
    VideoUploadResponse,
)
from config import get_settings

settings = get_settings()
router = APIRouter(prefix="/video", tags=["video"])
logger = logger.bind(name="VideoRouter")


# ── Task status enum ───────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    FAILED      = "failed"
    NOT_FOUND   = "not_found"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=VideoUploadResponse, summary="Upload a video file")
async def upload_video(file: UploadFile = File(...)) -> VideoUploadResponse:
    """
    Accept a multipart video upload, save it to SHARED_MEDIA_DIR, and
    return the server-side path.

    The path returned here is what you pass to `/video/process` and to
    the `/chat` endpoint as `video_path`.

    Uploading the same filename twice is idempotent — the existing file
    is reused without re-uploading.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    media_dir = Path(settings.SHARED_MEDIA_DIR)
    media_dir.mkdir(parents=True, exist_ok=True)

    dest = media_dir / file.filename

    if not dest.exists():
        try:
            with open(dest, "wb") as fh:
                shutil.copyfileobj(file.file, fh)
            logger.info(f"Saved upload → '{dest}'")
        except Exception as exc:
            logger.error(f"Upload failed for '{file.filename}': {exc}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")
    else:
        logger.info(f"File '{dest}' already exists — reusing.")

    return VideoUploadResponse(
        message="Video uploaded successfully.",
        video_path=str(dest),
    )


@router.post(
    "/process",
    response_model=ProcessVideoResponse,
    summary="Ingest a video (background task)",
)
async def process_video(
    request: ProcessVideoRequest,
    fastapi_request: Request,
) -> ProcessVideoResponse:
    """
    Kick off the full ingestion pipeline for a video file as a background task:
    - Extract frames + audio
    - Transcribe audio with AWS Transcribe / Whisper
    - Caption scenes with Nova Pro (batched)
    - Build multimodal embedding indexes via Amazon Titan

    Returns a `task_id` immediately.  Poll `GET /video/task-status/{task_id}`
    to track progress.
    """
    if not Path(request.video_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found: '{request.video_path}'",
        )

    task_id = uuid4().hex
    task_states: dict = fastapi_request.app.state.task_states
    task_states[task_id] = TaskStatus.PENDING

    async def _run(video_path: str, task_id: str) -> None:
        """Background task to process video via MCP tool."""
        nonlocal task_states
        try:
            task_states[task_id] = TaskStatus.IN_PROGRESS
            start_time = time.perf_counter()
            logger.info(f"Task {task_id}: Starting video ingestion for '{video_path}'")
            
            from fastmcp import Client
            logger.debug(f"Task {task_id}: Connecting to MCP server at {settings.MCP_SERVER_URL}")
            
            async with Client(settings.MCP_SERVER_URL) as client:
                logger.debug(f"Task {task_id}: Connected to MCP server, calling process_video tool")
                result = await client.call_tool("process_video", {"video_path": video_path})
                logger.info(f"Task {task_id}: MCP tool returned successfully")
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            task_states[task_id] = TaskStatus.COMPLETED
            logger.info(f"Task {task_id}: ✅ Ingestion COMPLETED for '{video_path}' | ⏱{duration_ms:.1f}ms")
        except Exception as exc:
            import traceback
            duration_ms = (time.perf_counter() - start_time) * 1000
            task_states[task_id] = TaskStatus.FAILED
            logger.error(f"Task {task_id}: ❌ Ingestion FAILED for '{video_path}' — {exc} | ⏱{duration_ms:.1f}ms")
            logger.error(f"Task {task_id}: Traceback: {traceback.format_exc()}")

    # Create background task and add a done callback to log any exceptions
    task = asyncio.create_task(_run(request.video_path, task_id))
    
    def _task_done_callback(t: asyncio.Task) -> None:
        if t.exception():
            logger.error(f"Task {task_id}: Background task raised exception: {t.exception()}")
    
    task.add_done_callback(_task_done_callback)
    
    logger.info(f"Task {task_id}: Background task created for '{request.video_path}'")

    return ProcessVideoResponse(
        message="Ingestion task queued.",
        task_id=task_id,
    )


@router.get(
    "/task-status/{task_id}",
    response_model=TaskStatusResponse,
    summary="Poll background task status",
)
async def task_status(task_id: str, fastapi_request: Request) -> TaskStatusResponse:
    """
    Returns the current status of a background ingestion task.

    Status values: `pending` | `in_progress` | `completed` | `failed` | `not_found`
    """
    task_states: dict = fastapi_request.app.state.task_states
    status = task_states.get(task_id, TaskStatus.NOT_FOUND)
    return TaskStatusResponse(task_id=task_id, status=status)


@router.get(
    "/media/{file_path:path}",
    summary="Serve a video clip or uploaded file",
)
async def serve_media(file_path: str) -> FileResponse:
    """
    Serve any file from SHARED_MEDIA_DIR.

    The `file_path` parameter is the filename only (path traversal is
    stripped) so callers cannot escape the media directory.
    """
    # Strip any directory component to prevent path traversal.
    safe_name = Path(file_path).name
    full_path = Path(settings.SHARED_MEDIA_DIR) / safe_name

    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: '{safe_name}'")

    return FileResponse(str(full_path))