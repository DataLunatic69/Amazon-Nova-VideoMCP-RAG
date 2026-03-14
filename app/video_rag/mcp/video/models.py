"""
Data models for the video index registry.

CachedTableMetadata  — serialisable snapshot stored in the JSON registry.
CachedTable          — live runtime object holding actual pxt.Table handles.
"""

import base64
import io
from typing import List

import pixeltable as pxt
from PIL import Image
from pydantic import BaseModel, Field


# ── Registry models ────────────────────────────────────────────────────────────

class CachedTableMetadata(BaseModel):
    """Serialisable record stored in the on-disk JSON registry."""

    video_name: str = Field(description="Logical key — typically the video file path.")
    video_cache: str = Field(description="Pixeltable directory name for this video.")
    video_table: str = Field(description="Fully-qualified name of the root video table.")
    frames_view: str = Field(description="Fully-qualified name of the frames view.")
    audio_chunks_view: str = Field(description="Fully-qualified name of the audio-chunks view.")


class CachedTable:
    """
    Runtime wrapper holding live pxt.Table / pxt.View references for one video.

    Constructed from a CachedTableMetadata via `CachedTable.from_metadata()`.
    """

    def __init__(
        self,
        video_name: str,
        video_cache: str,
        video_table: pxt.Table,
        frames_view: pxt.Table,
        audio_chunks_view: pxt.Table,
    ) -> None:
        self.video_name = video_name
        self.video_cache = video_cache
        self.video_table = video_table
        self.frames_view = frames_view
        self.audio_chunks_view = audio_chunks_view

    @classmethod
    def from_metadata(cls, metadata: "CachedTableMetadata | dict") -> "CachedTable":
        if isinstance(metadata, dict):
            metadata = CachedTableMetadata(**metadata)
        return cls(
            video_name=metadata.video_name,
            video_cache=metadata.video_cache,
            video_table=pxt.get_table(metadata.video_table),
            frames_view=pxt.get_table(metadata.frames_view),
            audio_chunks_view=pxt.get_table(metadata.audio_chunks_view),
        )

    def describe(self) -> str:
        cols = ", ".join(str(c) for c in self.video_table.columns)
        return f"Video index '{self.video_name}' — columns: {cols}"

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CachedTable(video_name={self.video_name!r}, "
            f"cache={self.video_cache!r})"
        )


# ── Image helpers ──────────────────────────────────────────────────────────────

class Base64Image(BaseModel):
    """Wraps a PIL Image as a base64 string for JSON serialisation."""

    image: str = Field(description="Base64-encoded JPEG string.")

    @classmethod
    def from_pil(cls, img: Image.Image) -> "Base64Image":
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return cls(image=base64.b64encode(buf.getvalue()).decode("utf-8"))

    def to_pil(self) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(self.image)))