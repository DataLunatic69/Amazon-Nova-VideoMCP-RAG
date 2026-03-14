"""
Pydantic models shared by the agent layer and the API layer.

Structured output models (suffixed *Schema) are what Nova Pro is instructed
to produce via JSON-mode — keeping the LLM response strongly typed avoids
ad-hoc string parsing throughout the codebase.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ── API request / response models ─────────────────────────────────────────────

class UserMessageRequest(BaseModel):
    message: str
    video_path: Optional[str] = None
    image_base64: Optional[str] = None


class AssistantMessageResponse(BaseModel):
    message: str
    clip_path: Optional[str] = None


class ResetMemoryResponse(BaseModel):
    message: str


# ── Memory ─────────────────────────────────────────────────────────────────────

class MemoryRecord(BaseModel):
    message_id: str
    role: Literal["user", "assistant"]
    content: str


# ── Structured LLM output schemas ─────────────────────────────────────────────

class RoutingSchema(BaseModel):
    """
    Output of the routing step.

    Nova Lite returns this to decide whether the user's message needs a
    tool call (clip search / QA) or can be answered with general chat.
    """
    tool_use: bool = Field(
        description=(
            "True if the user's message requires operating on the video "
            "(finding a clip, answering a factual question, or image search). "
            "False for general conversation."
        )
    )


class ToolSelectionSchema(BaseModel):
    """
    Output of the tool-selection step.

    Nova Pro returns this to pick exactly one tool and supply its arguments.
    """
    tool_name: Literal[
        "get_video_clip_from_query",
        "get_video_clip_from_image",
        "ask_question_about_video",
        "get_video_clip_by_time",
    ] = Field(description="The tool to call.")

    user_query: Optional[str] = Field(
        default=None,
        description="The search query or question to pass to the tool.",
    )
    start_seconds: Optional[float] = Field(
        default=None,
        description="Start time in seconds (for get_video_clip_by_time).",
    )
    end_seconds: Optional[float] = Field(
        default=None,
        description="End time in seconds (for get_video_clip_by_time).",
    )


class GeneralResponseSchema(BaseModel):
    """Nova Pro general conversational response."""
    message: str = Field(description="Response to the user.")


class VideoClipResponseSchema(BaseModel):
    """Nova Pro response after a clip has been retrieved."""
    message: str = Field(
        description=(
            "An engaging message telling the user what was found and "
            "inviting them to watch the clip."
        )
    )
    clip_path: str = Field(description="Filesystem path to the generated clip.")


class QAResponseSchema(BaseModel):
    """Nova Pro response after RAG answer generation."""
    message: str = Field(
        description="The grounded answer to the user's question about the video."
    )