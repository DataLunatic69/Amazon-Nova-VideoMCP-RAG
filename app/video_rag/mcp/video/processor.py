"""
VideoProcessor — builds and populates the Pixeltable index for one video.

Pipeline (per video)
────────────────────
Video file
  │
  ├─ audio_extract        (pixeltable built-in)
  │     └─ audio_chunks   (AudioSplitter view)
  │           ├─ transcription  (AWS Transcribe or Whisper)
  │           ├─ chunk_text     (UDF: extract plain text)
  │           └─ embedding      (Titan Multimodal — TEXT path)
  │                 └─ vector index "speech_index"
  │
  └─ frames               (FrameIterator view)
        ├─ resized_frame  (UDF: thumbnail)
        ├─ embedding      (Titan Multimodal — IMAGE path)  ← unified space
        │     └─ vector index "frame_index"
        ├─ scene_caption  (Nova Pro — batched scene understanding)
        └─ caption_embed  (Titan Multimodal — TEXT path)
              └─ vector index "caption_index"

Key difference from the reference implementation
─────────────────────────────────────────────────
All three vector indexes use Amazon Titan Multimodal Embeddings, which
projects text AND images into a *shared* vector space.  This means a
plain text query can be compared directly against image embeddings —
no separate CLIP model required.
"""

import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pixeltable as pxt
from pixeltable.functions.video import extract_audio, legacy_frame_iterator
from pixeltable.functions.audio import audio_splitter

import video_rag.mcp.video.registry as registry
from video_rag.config import get_settings
from video_rag.mcp.video.functions import extract_transcript_text, resize_image
from video_rag.mcp.video.media import re_encode_video
from logging_config import get_logger

if TYPE_CHECKING:
    from video_rag.mcp.video.models import CachedTable

settings = get_settings()
logger = get_logger("VideoProcessor")


# ── Pixeltable UDF wrappers for Nova embeddings ────────────────────────────────
# We define these at module level so Pixeltable can serialise them.

@pxt.udf
def _embed_text_udf(text: pxt.String) -> pxt.Array[(1024,), pxt.Float]:
    """Titan Multimodal text embedding — called inside a computed column."""
    from video_rag.mcp.video.nova_client import embed_text
    return embed_text(text)


@pxt.udf
def _embed_image_udf(image: pxt.Image) -> pxt.Array[(1024,), pxt.Float]:
    """Titan Multimodal image embedding — called inside a computed column."""
    from video_rag.mcp.video.nova_client import embed_image
    return embed_image(image)


@pxt.udf
def _caption_scene_udf(frame: pxt.Image) -> pxt.String:
    """
    Nova Pro scene captioning for a single video frame.

    Wraps the frame in a list so ``caption_scene()`` can be extended to
    accept batches in the future without changing the call signature.
    TODO: use a Pixeltable window UDF to batch SCENE_CAPTION_BATCH_SIZE
          consecutive frames into one Nova Pro call for richer context.
    """
    from video_rag.mcp.video.nova_client import caption_scene

    return caption_scene([frame])


@pxt.udf
def _transcribe_audio_udf(audio: pxt.Audio) -> pxt.Json:
    """
    Transcribe an audio chunk.  Dispatches to AWS Transcribe or Whisper
    depending on settings.USE_AWS_TRANSCRIBE.
    """
    if settings.USE_AWS_TRANSCRIBE:
        from video_rag.mcp.video.transcription import transcribe_with_aws
        return transcribe_with_aws(audio)
    else:
        from video_rag.mcp.video.transcription import transcribe_with_whisper
        return transcribe_with_whisper(audio)


# ── Main class ─────────────────────────────────────────────────────────────────

class VideoProcessor:
    """Manages Pixeltable table/view creation for a single video."""

    def __init__(self) -> None:
        self._video_mapping_idx: Optional[str] = None
        self.pxt_cache: Optional[str] = None
        self.video_table_name: Optional[str] = None
        self.frames_view_name: Optional[str] = None
        self.audio_view_name: Optional[str] = None

        self.video_table = None
        self.frames_view = None
        self.audio_chunks = None

        logger.info(
            f"VideoProcessor ready. "
            f"frames={settings.SPLIT_FRAMES_COUNT}, "
            f"chunk={settings.AUDIO_CHUNK_LENGTH}s, "
            f"transcription={'aws' if settings.USE_AWS_TRANSCRIBE else 'whisper'}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def already_indexed(self, video_name: str) -> bool:
        return video_name in registry.get_registry()

    def setup_index(self, video_name: str) -> None:
        """Create Pixeltable directory, tables, views, and indexes."""
        self._video_mapping_idx = video_name

        if self.already_indexed(video_name):
            self._load_existing(video_name)
            return

        self.pxt_cache = f"vr_{uuid.uuid4().hex[:6]}"
        self.video_table_name = f"{self.pxt_cache}.table"
        self.frames_view_name = f"{self.video_table_name}_frames"
        self.audio_view_name = f"{self.video_table_name}_audio"

        self._build_pipeline()

        registry.add_index_to_registry(
            video_name=video_name,
            video_cache=self.pxt_cache,
            frames_view_name=self.frames_view_name,
            audio_view_name=self.audio_view_name,
        )
        logger.info(f"Index '{self.video_table_name}' created for '{video_name}'.")

    def add_video(self, video_path: str) -> bool:
        """Insert the video row and trigger all computed columns."""
        if self.video_table is None:
            raise RuntimeError("Call setup_index() before add_video().")

        safe_path = re_encode_video(video_path)
        if not safe_path:
            logger.error(f"Could not prepare '{video_path}' for ingestion.")
            return False

        self.video_table.insert([{"video": safe_path}])
        logger.info(f"Inserted '{safe_path}' — Pixeltable is computing columns …")
        return True

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_existing(self, video_name: str) -> None:
        """Reattach to existing Pixeltable objects for an already-indexed video."""
        cached: "CachedTable" = registry.get_table(video_name)
        self.pxt_cache = cached.video_cache
        self.video_table = cached.video_table
        self.frames_view = cached.frames_view
        self.audio_chunks = cached.audio_chunks_view
        logger.info(f"Reattached to existing index for '{video_name}'.")

    def _build_pipeline(self) -> None:
        self._create_directory()
        self._create_video_table()
        self._build_audio_pipeline()
        self._build_frame_pipeline()

    def _create_directory(self) -> None:
        Path(self.pxt_cache).mkdir(parents=True, exist_ok=True)
        pxt.create_dir(self.pxt_cache, if_exists="replace_force")

    def _create_video_table(self) -> None:
        self.video_table = pxt.create_table(
            self.video_table_name,
            schema={"video": pxt.Video},
            if_exists="replace_force",
        )

    # ── Audio pipeline ─────────────────────────────────────────────────────────

    def _build_audio_pipeline(self) -> None:
        # Step 1: extract mp3 from the video.
        self.video_table.add_computed_column(
            audio_extract=extract_audio(self.video_table.video, format="mp3"),
            if_exists="ignore",
        )

        # Step 2: split audio into overlapping chunks.
        self.audio_chunks = pxt.create_view(
            self.audio_view_name,
            self.video_table,
            iterator=audio_splitter(
                self.video_table.audio_extract,
                duration=settings.AUDIO_CHUNK_LENGTH,
                overlap=settings.AUDIO_OVERLAP_SECONDS,
                min_segment_duration=settings.AUDIO_MIN_CHUNK_DURATION_SECONDS,
            ),
            if_exists="replace_force",
        )

        # Step 3: transcribe each chunk.
        self.audio_chunks.add_computed_column(
            transcription=_transcribe_audio_udf(self.audio_chunks.audio_segment),
            if_exists="ignore",
        )

        # Step 4: extract plain text from the transcription JSON.
        self.audio_chunks.add_computed_column(
            chunk_text=extract_transcript_text(self.audio_chunks.transcription),
            if_exists="ignore",
        )

        # Step 5: build a text-embedding index directly on chunk_text.
        self.audio_chunks.add_embedding_index(
            self.audio_chunks.chunk_text,
            embedding=_embed_text_udf,
            metric="cosine",
            if_exists="ignore",
            idx_name="speech_index",
        )

    # ── Frame pipeline ─────────────────────────────────────────────────────────

    def _build_frame_pipeline(self) -> None:
        # Step 1: extract uniformly-sampled frames.
        self.frames_view = pxt.create_view(
            self.frames_view_name,
            self.video_table,
            iterator=legacy_frame_iterator(
                self.video_table.video,
                num_frames=settings.SPLIT_FRAMES_COUNT,
            ),
            if_exists="ignore",
        )

        # Step 2: resize for cost/performance.
        self.frames_view.add_computed_column(
            resized_frame=resize_image(
                self.frames_view.frame,
                width=settings.IMAGE_RESIZE_WIDTH,
                height=settings.IMAGE_RESIZE_HEIGHT,
            ),
            if_exists="ignore",
        )

        # Step 3: build an image-embedding index directly on resized_frame.
        self.frames_view.add_embedding_index(
            self.frames_view.resized_frame,
            embedding=_embed_image_udf,
            metric="cosine",
            if_exists="ignore",
            idx_name="frame_index",
        )

        # Step 4: scene captioning with Nova Pro.
        self.frames_view.add_computed_column(
            scene_caption=_caption_scene_udf(self.frames_view.resized_frame),
            if_exists="ignore",
        )

        # Step 5: build a text-embedding index directly on scene_caption.
        self.frames_view.add_embedding_index(
            self.frames_view.scene_caption,
            embedding=_embed_text_udf,
            metric="cosine",
            if_exists="ignore",
            idx_name="caption_index",
        )