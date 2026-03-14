"""
VideoSearchEngine — similarity search across all three indexes.

All three indexes (speech, frame, caption) use Amazon Titan Multimodal
Embeddings, so a single text query can be compared against image embeddings
directly.  This is the key architectural improvement over the reference
implementation which used separate CLIP + text-embedding models.

Search methods
──────────────
search_by_speech(query)  → top-k audio transcript chunks ranked by cosine sim
search_by_caption(query) → top-k scene captions ranked by cosine sim
search_by_image(image)   → top-k frames ranked by cosine sim to a query image

Context methods (used by ask_question_about_video)
───────────────────────────────────────────────────
get_speech_context(query)  → transcript text + similarity score dicts
get_caption_context(query) → caption text + similarity score dicts
"""

import time
from typing import Any, Dict, List, Optional

import video_rag.mcp.video.registry as registry
from video_rag.config import get_settings
from video_rag.mcp.video.models import CachedTable
from logging_config import get_logger


settings = get_settings()
logger = get_logger("SearchEngine")


class VideoSearchEngine:
    """Similarity search over a processed video index."""

    def __init__(self, video_name: str) -> None:
        self.video_name = video_name
        self._index: Optional[CachedTable] = registry.get_table(video_name)
        if self._index is None:
            raise ValueError(
                f"No index found for '{video_name}'. "
                "Run process_video() first."
            )

    # ── Speech search ──────────────────────────────────────────────────────────

    def search_by_speech(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Find transcript chunks semantically similar to *query*.

        Returns list of dicts:  {start_time, end_time, similarity, text}
        """
        audio = self._index.audio_chunks_view
        
        if audio is None:
            logger.warning(f"No audio chunks found for '{self.video_name}'")
            return []

        try:
            sims = audio.chunk_text.similarity(string=query)
            # Collect all rows, then filter None values in Python
            # (Pixeltable does not support .is_not_null() on similarity expressions)
            rows = (
                audio.select(
                    audio.segment_start,
                    audio.segment_end,
                    audio.chunk_text,
                    similarity=sims,
                )
                .collect()
            )

            results = []
            for r in rows:
                if r.get("similarity") is None or r.get("chunk_text") is None:
                    continue
                results.append({
                    "start_time": float(r["segment_start"] or 0),
                    "end_time": float(r["segment_end"] or 0),
                    "similarity": float(r["similarity"]),
                    "text": r["chunk_text"],
                })
            # Sort by similarity descending and take top_k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Speech search failed for '{self.video_name}': {e}")
            return []

    # ── Caption search ─────────────────────────────────────────────────────────

    def search_by_caption(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Find scene captions semantically similar to *query*.

        Returns list of dicts:  {start_time, end_time, similarity, text}
        """
        frames = self._index.frames_view
        
        if frames is None:
            logger.warning(f"No frames found for '{self.video_name}'")
            return []

        sims = frames.scene_caption.similarity(string=query)
        rows = (
            frames.select(
                frames.pos_msec,
                frames.scene_caption,
                similarity=sims,
            )
            .order_by(sims, asc=False)
            .limit(top_k)
            .collect()
        )

        results = []
        for r in rows:
            # Skip rows with missing data
            if r.get("similarity") is None or r.get("scene_caption") is None:
                continue
            pos_msec = r.get("pos_msec") or 0
            results.append({
                "start_time": pos_msec / 1000.0 - settings.DELTA_SECONDS_FRAME_INTERVAL,
                "end_time": pos_msec / 1000.0 + settings.DELTA_SECONDS_FRAME_INTERVAL,
                "similarity": float(r["similarity"]),
                "text": r["scene_caption"],
            })
        return results

    # ── Image search ───────────────────────────────────────────────────────────

    def search_by_image(self, image_b64: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Find frames visually similar to a base64-encoded query image.

        Because both query and index use Titan Multimodal Embeddings, this
        cross-modal comparison works in the same vector space.

        Returns list of dicts:  {start_time, end_time, similarity}
        """
        import base64
        import io
        from PIL import Image as PILImage

        frames = self._index.frames_view
        
        if frames is None:
            logger.warning(f"No frames found for '{self.video_name}'")
            return []

        try:
            raw = base64.b64decode(image_b64)
            pil_image = PILImage.open(io.BytesIO(raw))

            sims = frames.resized_frame.similarity(image=pil_image)
            # Collect all rows, then filter None values in Python
            rows = (
                frames.select(
                    frames.pos_msec,
                    similarity=sims,
                )
                .collect()
            )

            results = []
            for r in rows:
                if r.get("similarity") is None:
                    continue
                pos_msec = r.get("pos_msec") or 0
                results.append({
                    "start_time": pos_msec / 1000.0 - settings.DELTA_SECONDS_FRAME_INTERVAL,
                    "end_time": pos_msec / 1000.0 + settings.DELTA_SECONDS_FRAME_INTERVAL,
                    "similarity": float(r["similarity"]),
                })
            # Sort by similarity descending and take top_k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Image search failed for '{self.video_name}': {e}")
            return []

    # ── Context helpers ────────────────────────────────────────────────────────

    def get_speech_context(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Return transcript chunks + similarity scores for RAG context."""
        return [
            {"text": r["text"], "similarity": r["similarity"]}
            for r in self.search_by_speech(query, top_k)
        ]

    def get_caption_context(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Return scene captions + similarity scores for RAG context."""
        return [
            {"text": r["text"], "similarity": r["similarity"]}
            for r in self.search_by_caption(query, top_k)
        ]