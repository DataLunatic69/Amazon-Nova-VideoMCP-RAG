"""
Chat router — text conversation with the NovaAgent.

Endpoints
─────────
POST   /chat           Send a text message (with optional video + image context).
DELETE /chat/memory    Reset conversation history for the current session.
GET    /chat/history   Retrieve the last N messages from memory.
"""

import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from video_rag.agent.models import (
    AssistantMessageResponse,
    MemoryRecord,
    ResetMemoryResponse,
    UserMessageRequest,
)
from logging_config import get_logger

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger("ChatRouter")


# ── POST /chat ─────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=AssistantMessageResponse,
    summary="Send a message to the agent",
)
async def chat(
    body: UserMessageRequest,
    request: Request,
) -> AssistantMessageResponse:
    """
    Send a text message to the NovaAgent and receive a response.

    **Routing logic:**
    - No `video_path` → general conversation (no tool calls, no video context).
    - `video_path` supplied → agent decides whether to search / QA or chat.
    - `video_path` + `image_base64` → forced visual similarity search.

    The video must have been ingested first via `POST /api/v1/video/process`.
    Use `GET /api/v1/video/task-status/{task_id}` to check ingestion progress.

    **Response:**
    - `message`: The agent's natural-language reply.
    - `clip_path`: Filesystem path to a generated clip (null when not applicable).
      Serve it with `GET /api/v1/video/media/{clip_path}`.
    """
    start_time = time.perf_counter()
    
    logger.info(
        "Chat endpoint called",
        message_length=len(body.message),
        has_video=bool(body.video_path),
        has_image=bool(body.image_base64),
        video_path=body.video_path,
    )
    
    agent = request.app.state.agent
    await agent.setup()

    try:
        response = await agent.chat(
            message=body.message,
            video_path=body.video_path,
            image_base64=body.image_base64,
        )
        
        duration = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Chat endpoint completed",
            response_length=len(response.message),
            has_clip=bool(response.clip_path),
            duration_ms=duration,
        )
        
    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000
        logger.error(
            "Chat endpoint failed",
            error=str(exc),
            error_type=type(exc).__name__,
            duration_ms=duration,
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return response


# ── DELETE /chat/memory ────────────────────────────────────────────────────────

@router.delete(
    "/memory",
    response_model=ResetMemoryResponse,
    summary="Reset conversation history",
)
async def reset_memory(request: Request) -> ResetMemoryResponse:
    """
    Wipe the agent's conversation memory, starting a fresh session.

    Useful when the user switches topics, loads a new video, or explicitly
    requests a fresh start.  The agent will re-fetch system prompts from
    the MCP server on the next request.
    """
    agent = request.app.state.agent
    agent.reset_memory()
    logger.info("Conversation memory cleared by user request.")
    return ResetMemoryResponse(message="Conversation memory cleared.")


# ── GET /chat/history ──────────────────────────────────────────────────────────

@router.get(
    "/history",
    summary="Get recent conversation history",
)
async def get_history(
    request: Request,
    n: int = Query(default=20, ge=1, le=200, description="Number of recent messages to return."),
):
    """
    Return the last *n* messages from the agent's memory.

    Useful for rendering conversation history in a UI.
    Messages are returned in chronological order (oldest first).
    """
    agent = request.app.state.agent
    records: List[MemoryRecord] = agent.memory.get_latest(n)
    return {
        "session_id": agent._session_id,
        "count": len(records),
        "messages": [
            {"role": r.role, "content": r.content, "message_id": r.message_id}
            for r in records
        ],
    }
