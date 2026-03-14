"""
On-disk registry of processed video indexes.

Every time a new video is indexed a timestamped JSON file is written to
REGISTRY_DIR. On startup the latest snapshot is loaded back into memory so
that the MCP server does not need to re-index videos across restarts.

Public API
----------
get_registry()          → Dict[str, CachedTableMetadata]
get_table(video_name)   → CachedTable
add_index_to_registry(…)
"""

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from video_rag.config import get_settings
from video_rag.mcp.video.models import CachedTable, CachedTableMetadata

settings = get_settings()
logger = logger.bind(name="Registry")

# In-process cache — avoids repeated disk reads within a single run.
_REGISTRY: Dict[str, CachedTableMetadata] = {}


@lru_cache(maxsize=1)
def get_registry() -> Dict[str, CachedTableMetadata]:
    """
    Load and return the video index registry.

    On the first call the latest JSON snapshot from REGISTRY_DIR is read.
    Subsequent calls return the in-memory cache.
    """
    global _REGISTRY
    if _REGISTRY:
        logger.debug("Returning in-memory registry.")
        return _REGISTRY

    registry_dir = Path(settings.REGISTRY_DIR)
    try:
        snapshots = sorted(
            f for f in os.listdir(registry_dir)
            if f.startswith("registry_") and f.endswith(".json")
        )
        if snapshots:
            latest = registry_dir / snapshots[-1]
            with open(latest) as fh:
                raw = json.load(fh)
            for key, value in raw.items():
                if isinstance(value, str):
                    value = json.loads(value)
                _REGISTRY[key] = CachedTableMetadata(**value)
            logger.info(f"Loaded {len(_REGISTRY)} video index(es) from '{latest}'.")
    except FileNotFoundError:
        logger.info(f"Registry directory '{registry_dir}' not found — starting fresh.")
    except Exception as exc:
        logger.warning(f"Failed to load registry: {exc}. Starting with empty registry.")

    return _REGISTRY


def get_table(video_name: str) -> Optional[CachedTable]:
    """Return a live CachedTable for *video_name*, or None if not registered."""
    registry = get_registry()
    metadata = registry.get(video_name)
    if metadata is None:
        return None
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return CachedTable.from_metadata(metadata)


def add_index_to_registry(
    video_name: str,
    video_cache: str,
    frames_view_name: str,
    audio_view_name: str,
) -> None:
    """
    Persist a newly created video index to the registry.

    Writes a timestamped JSON snapshot and updates the in-memory cache.
    """
    global _REGISTRY

    # Invalidate the lru_cache so the next get_registry() re-reads from _REGISTRY.
    get_registry.cache_clear()

    meta = CachedTableMetadata(
        video_name=video_name,
        video_cache=video_cache,
        video_table=f"{video_cache}.table",
        frames_view=frames_view_name,
        audio_chunks_view=audio_view_name,
    )
    _REGISTRY[video_name] = meta

    # Write snapshot to disk.
    registry_dir = Path(settings.REGISTRY_DIR)
    registry_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = registry_dir / f"registry_{timestamp}.json"

    serialised = {
        k: v.model_dump_json() if isinstance(v, CachedTableMetadata) else v
        for k, v in _REGISTRY.items()
    }
    with open(snapshot_path, "w") as fh:
        json.dump(serialised, fh, indent=2)

    logger.info(f"Registry snapshot written to '{snapshot_path}'.")