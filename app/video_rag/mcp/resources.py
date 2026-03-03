"""
MCP resources — read-only data exposed to MCP clients.
"""

from typing import Dict, Optional

from video_rag.mcp.video.registry import get_registry


def list_videos() -> Optional[Dict]:
    """
    Return all videos that have been processed and indexed.

    Exposed as an MCP resource at ``file:///app/.records/records.json``.
    """
    keys = list(get_registry().keys())
    if not keys:
        return None
    return {
        "message": "Processed videos ready for search",
        "videos": keys,
        "count": len(keys),
    }