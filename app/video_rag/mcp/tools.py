"""
MCP-exposed tools.

Each function is registered on the FastMCP server in server.py.
They are thin orchestration wrappers: they delegate all heavy work to
VideoProcessor (ingestion) and VideoSearchEngine (retrieval), then use
ffmpeg via extract_video_clip() to materialise the output clip on disk.
"""

import time
from pathlib import Path
from uuid import uuid4

from config import get_settings
from logging_config import get_logger, log_performance
from video_rag.mcp.video.processor import VideoProcessor
from video_rag.mcp.video.search_engine import VideoSearchEngine
from video_rag.mcp.video.media import extract_video_clip

settings = get_settings()
logger = get_logger("MCPTools")

# Single processor instance per MCP process lifetime.
_processor = VideoProcessor()


# ── Ingestion ──────────────────────────────────────────────────────────────────

async def process_video(video_path: str) -> str:
    """
    Ingest a video: extract frames + audio, transcribe, caption scenes with
    Nova Pro, build unified multimodal embedding indexes via Amazon Titan.

    Args:
        video_path: Absolute or relative path to the video file.

    Returns:
        Human-readable status string.
    """
    start_time = time.perf_counter()
    
    logger.info(
        "Video ingestion started",
        video_path=video_path,
        file_exists=Path(video_path).exists(),
    )
    
    if not Path(video_path).exists():
        logger.error("Video file not found", video_path=video_path)
        return f"Error: file not found at '{video_path}'."

    if _processor.already_indexed(video_path):
        logger.info(
            "Video already indexed, skipping",
            video_path=video_path,
        )
        return f"Video '{video_path}' was already processed and is ready to search."

    logger.info(
        "Starting full video ingestion pipeline",
        video_path=video_path,
        frames_count=settings.SPLIT_FRAMES_COUNT,
        chunk_length_sec=settings.AUDIO_CHUNK_LENGTH,
    )
    
    _processor.setup_index(video_name=video_path)
    _processor.add_video(video_path=video_path)
    
    duration = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Video ingestion completed",
        video_path=video_path,
        duration_ms=duration,
    )

    return f"Video '{video_path}' processed successfully and is ready to search."


# ── Clip retrieval ─────────────────────────────────────────────────────────────

async def get_video_clip_from_query(video_path: str, user_query: str) -> str:
    """
    Return a path to the clip that best matches *user_query*.

    Strategy: run speech-similarity search AND caption-similarity search in
    parallel, then pick the result with the higher similarity score.

    Args:
        video_path:  Path to the already-processed video.
        user_query:  Free-text query from the user.

    Returns:
        Path to the trimmed .mp4 clip.
    """
    start_time = time.perf_counter()
    
    logger.info(
        "Clip search started",
        video_path=video_path,
        query=user_query[:100] + "..." if len(user_query) > 100 else user_query,
    )
    
    engine = VideoSearchEngine(video_path)

    speech_start = time.perf_counter()
    speech_hits = engine.search_by_speech(user_query, top_k=settings.SPEECH_SEARCH_TOP_K)
    speech_duration = (time.perf_counter() - speech_start) * 1000
    
    caption_start = time.perf_counter()
    caption_hits = engine.search_by_caption(user_query, top_k=settings.CAPTION_SEARCH_TOP_K)
    caption_duration = (time.perf_counter() - caption_start) * 1000
    
    logger.debug(
        "Search results",
        speech_hits=len(speech_hits),
        caption_hits=len(caption_hits),
        speech_duration_ms=speech_duration,
        caption_duration_ms=caption_duration,
        best_speech_sim=speech_hits[0]["similarity"] if speech_hits else 0,
        best_caption_sim=caption_hits[0]["similarity"] if caption_hits else 0,
    )

    # Guard against empty results from either modality.
    best_speech = speech_hits[0] if speech_hits else {"similarity": 0.0}
    best_caption = caption_hits[0] if caption_hits else {"similarity": 0.0}

    best = best_speech if best_speech["similarity"] >= best_caption["similarity"] else best_caption
    modality = "speech" if best_speech["similarity"] >= best_caption["similarity"] else "caption"

    if best["similarity"] < settings.SIMILARITY_THRESHOLD:
        logger.warning(
            "No clip found above similarity threshold",
            best_similarity=best["similarity"],
            threshold=settings.SIMILARITY_THRESHOLD,
            query=user_query[:50],
        )
        return "No sufficiently relevant clip found. Try rephrasing your query."

    output_path = str(Path(settings.SHARED_MEDIA_DIR) / f"clip_{uuid4().hex[:8]}.mp4")
    
    logger.info(
        "Extracting clip",
        modality=modality,
        similarity=best["similarity"],
        start_time=best["start_time"],
        end_time=best["end_time"],
        output_path=output_path,
    )
    
    clip = extract_video_clip(
        video_path=video_path,
        start_time=best["start_time"],
        end_time=best["end_time"],
        output_path=output_path,
    )
    
    total_duration = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Clip extraction completed",
        clip_path=clip.filename,
        total_duration_ms=total_duration,
    )
    
    return clip.filename


async def get_video_clip_from_image(video_path: str, user_image: str) -> str:
    """
    Return the clip most visually similar to *user_image*.

    Args:
        video_path:  Path to the already-processed video.
        user_image:  Base64-encoded JPEG/PNG query image.

    Returns:
        Path to the trimmed .mp4 clip.
    """
    start_time = time.perf_counter()
    
    logger.info(
        "Image-based clip search started",
        video_path=video_path,
        image_length=len(user_image),
    )
    
    engine = VideoSearchEngine(video_path)
    hits = engine.search_by_image(user_image, top_k=settings.IMAGE_SEARCH_TOP_K)

    if not hits:
        logger.warning("No visually similar clips found", video_path=video_path)
        return "No visually similar clip found."

    best = hits[0]
    output_path = str(Path(settings.SHARED_MEDIA_DIR) / f"clip_{uuid4().hex[:8]}.mp4")
    
    logger.info(
        "Extracting image-matched clip",
        similarity=best["similarity"],
        start_time=best["start_time"],
        end_time=best["end_time"],
    )
    
    clip = extract_video_clip(
        video_path=video_path,
        start_time=best["start_time"],
        end_time=best["end_time"],
        output_path=output_path,
    )
    
    duration = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Image-based clip extraction completed",
        clip_path=clip.filename,
        duration_ms=duration,
    )
    
    return clip.filename


# ── Time-based clip extraction ────────────────────────────────────────────────

async def get_video_clip_by_time(video_path: str, start_seconds: float, end_seconds: float) -> str:
    """
    Extract a clip from a specific time range in the video.

    Use this when the user asks for a clip at a specific time
    (e.g. "first 4 seconds", "from 10 to 20 seconds", "the last 5 seconds").

    Args:
        video_path:    Path to the video file.
        start_seconds: Start time of the clip in seconds.
        end_seconds:   End time of the clip in seconds.

    Returns:
        Path to the trimmed .mp4 clip.
    """
    start_time = time.perf_counter()

    logger.info(
        "Time-based clip extraction started",
        video_path=video_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )

    if not Path(video_path).exists():
        return f"Error: video file not found at '{video_path}'."

    if start_seconds < 0:
        start_seconds = 0.0
    if end_seconds <= start_seconds:
        return f"Error: end_seconds ({end_seconds}) must be greater than start_seconds ({start_seconds})."

    output_path = str(Path(settings.SHARED_MEDIA_DIR) / f"clip_{uuid4().hex[:8]}.mp4")

    clip = extract_video_clip(
        video_path=video_path,
        start_time=start_seconds,
        end_time=end_seconds,
        output_path=output_path,
    )

    duration = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Time-based clip extraction completed",
        clip_path=clip.filename,
        duration_ms=duration,
    )

    return clip.filename


# ── QA ─────────────────────────────────────────────────────────────────────────

async def ask_question_about_video(video_path: str, user_query: str) -> str:
    """
    Answer a question about the video using RAG.

    Retrieves the most relevant transcript chunks + scene captions, then
    passes them as context to Amazon Nova Pro for answer generation.

    Args:
        video_path:  Path to the already-processed video.
        user_query:  Question from the user.

    Returns:
        Grounded natural-language answer.
    """
    start_time = time.perf_counter()
    
    logger.info(
        "RAG QA started",
        video_path=video_path,
        query=user_query[:100] + "..." if len(user_query) > 100 else user_query,
    )
    
    from video_rag.mcp.video.nova_client import generate_rag_answer

    engine = VideoSearchEngine(video_path)

    # Collect context from both modalities.
    speech_start = time.perf_counter()
    speech_ctx = engine.get_speech_context(user_query, top_k=settings.QA_CONTEXT_TOP_K)
    speech_duration = (time.perf_counter() - speech_start) * 1000
    
    caption_start = time.perf_counter()
    caption_ctx = engine.get_caption_context(user_query, top_k=settings.QA_CONTEXT_TOP_K)
    caption_duration = (time.perf_counter() - caption_start) * 1000

    # Deduplicate and rank by similarity (highest first).
    all_ctx = sorted(speech_ctx + caption_ctx, key=lambda x: x["similarity"], reverse=True)
    top_ctx = all_ctx[: settings.QA_CONTEXT_TOP_K]
    
    logger.debug(
        "Context retrieval completed",
        speech_chunks=len(speech_ctx),
        caption_chunks=len(caption_ctx),
        top_ctx_count=len(top_ctx),
        speech_duration_ms=speech_duration,
        caption_duration_ms=caption_duration,
        best_similarity=top_ctx[0]["similarity"] if top_ctx else 0,
    )

    if not top_ctx:
        logger.warning(
            "No relevant context found for RAG",
            video_path=video_path,
            query=user_query[:50],
        )
        return "I couldn't find relevant information in the video to answer that question."

    context_text = "\n\n".join(
        f"[{i+1}] (similarity={c['similarity']:.2f}) {c['text']}"
        for i, c in enumerate(top_ctx)
    )
    
    logger.debug(
        "Context prepared for RAG",
        context_length=len(context_text),
        chunks_used=len(top_ctx),
    )

    gen_start = time.perf_counter()
    answer = generate_rag_answer(question=user_query, context=context_text)
    gen_duration = (time.perf_counter() - gen_start) * 1000
    
    total_duration = (time.perf_counter() - start_time) * 1000
    logger.info(
        "RAG QA completed",
        answer_length=len(answer),
        context_chunks=len(top_ctx),
        generation_duration_ms=gen_duration,
        total_duration_ms=total_duration,
    )
    
    return answer