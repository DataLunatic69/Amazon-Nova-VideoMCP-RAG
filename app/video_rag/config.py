"""
Re-export of Settings / get_settings for code inside the video_rag package.

Several modules under video_rag/ use:

    from video_rag.config import get_settings

The canonical definition lives one level up at ``app/config.py``
(importable as plain ``config`` when ``app/`` is on sys.path).
This shim makes both import styles work without duplication.
"""

from config import Settings, get_settings  # noqa: F401

__all__ = ["Settings", "get_settings"]
