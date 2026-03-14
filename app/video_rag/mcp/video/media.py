"""
Media helpers — video clip extraction and compatibility re-encoding.

Both functions shell out to ffmpeg, which must be available in PATH.
"""

import subprocess
from pathlib import Path

from loguru import logger
from moviepy import VideoFileClip

logger = logger.bind(name="Media")


def extract_video_clip(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
) -> VideoFileClip:
    """
    Trim *video_path* from *start_time* to *end_time* and write to *output_path*.

    Uses ffmpeg directly (not MoviePy) for reliability with long videos.
    The start/end times are clamped so they never exceed the file duration.

    Args:
        video_path:   Source video file.
        start_time:   Clip start in seconds.
        end_time:     Clip end in seconds.
        output_path:  Destination .mp4 path.

    Returns:
        MoviePy VideoFileClip handle for the output file.

    Raises:
        ValueError:  If start_time >= end_time.
        IOError:     If ffmpeg fails.
    """
    start_time = max(0.0, start_time)
    if start_time >= end_time:
        raise ValueError(
            f"start_time ({start_time}) must be less than end_time ({end_time})."
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        "-y",                   # overwrite without prompting
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.debug(f"ffmpeg stdout: {result.stdout}")
    except subprocess.CalledProcessError as exc:
        raise IOError(f"ffmpeg clip extraction failed: {exc.stderr}") from exc

    return VideoFileClip(output_path)


def re_encode_video(video_path: str) -> str | None:
    """
    Ensure *video_path* can be opened by PyAV / Pixeltable.

    Some downloaded videos have container issues that PyAV cannot handle.
    This function attempts to open the file first; if that fails it re-encodes
    using ``ffmpeg -c copy`` (stream copy — fast, lossless) and returns the
    new path.

    Returns:
        Path to a PyAV-compatible video, or None if all attempts fail.
    """
    import av

    video_path = str(video_path)

    if not Path(video_path).exists():
        logger.error(f"Video not found: '{video_path}'")
        return None

    # Try opening as-is first.
    try:
        with av.open(video_path):
            pass
        return video_path
    except Exception as exc:
        logger.warning(f"PyAV could not open '{video_path}': {exc}. Re-encoding …")

    # Re-encode with stream copy.
    p = Path(video_path)
    reencoded = str(p.parent / f"re_{p.name}")
    cmd = ["ffmpeg", "-i", video_path, "-c", "copy", "-y", reencoded]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error(f"Re-encoding failed: {exc.stderr}")
        return None

    try:
        with av.open(reencoded):
            pass
        logger.info(f"Re-encoded video saved to '{reencoded}'.")
        return reencoded
    except Exception as exc:
        logger.error(f"Re-encoded file still unreadable: {exc}")
        return None