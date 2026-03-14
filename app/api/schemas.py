"""
API-layer Pydantic models.

Request/response models for endpoints that are not covered by the shared
agent models (agent/models.py).  Kept separate so the agent layer has zero
dependency on FastAPI.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Video ingestion ────────────────────────────────────────────────────────────

class ProcessVideoRequest(BaseModel):
    video_path: str = Field(description="Absolute or relative path to the video file.")


class ProcessVideoResponse(BaseModel):
    message: str
    task_id: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str   # pending | in_progress | completed | failed | not_found


class VideoUploadResponse(BaseModel):
    message: str
    video_path: str


# ── Voice REST ─────────────────────────────────────────────────────────────────

class VoiceChatResponse(BaseModel):
    """
    Response from POST /voice/chat.

    The audio bytes are returned as a separate streaming response;
    this JSON envelope is sent as a trailing header or via a second
    response depending on the client implementation.
    """
    transcript: str = Field(description="What the user said (STT output).")
    message: str    = Field(description="Agent text response.")
    clip_path: Optional[str] = Field(
        default=None,
        description="Path to generated clip, if any.",
    )
    audio_url: Optional[str] = Field(
        default=None,
        description="URL to the synthesised audio response file.",
    )