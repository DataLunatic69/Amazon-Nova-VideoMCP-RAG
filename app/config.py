from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives at <project_root>/app/config.py
# .env lives at <project_root>/.env — use an absolute path so pydantic-settings
# always finds it regardless of the CWD when the process is started.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = str(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        extra="ignore",
        env_file_encoding="utf-8",
    )

    # ── AWS / Amazon Nova ──────────────────────────────────────────────────────
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"

    # Nova model IDs (Bedrock)
    NOVA_LITE_MODEL_ID: str = "amazon.nova-lite-v1:0"
    NOVA_PRO_MODEL_ID: str = "amazon.nova-pro-v1:0"

    # Nova 2 Sonic — real-time bidirectional speech via Bedrock streaming.
    # Uses a separate WebSocket-based API (not Converse).
    NOVA_SONIC_MODEL_ID: str = "amazon.nova-sonic-v1:0"
    SONIC_VOICE_ID: str = "tiffany"          # Options: tiffany, matthew, amy
    SONIC_SAMPLE_RATE: int = 16000           # Hz — input audio sample rate
    SONIC_OUTPUT_SAMPLE_RATE: int = 24000    # Hz — output audio sample rate
    SONIC_AUDIO_INPUT_FORMAT: str = "audio/lpcm"
    SONIC_AUDIO_OUTPUT_FORMAT: str = "audio/lpcm"

    # Amazon Titan multimodal embeddings — single unified embedding space for
    # both text queries and images, replacing the 3 separate indexes in the
    # reference implementation (CLIP for frames, text-embedding for transcripts,
    # text-embedding for captions).
    NOVA_MULTIMODAL_EMBED_MODEL_ID: str = "amazon.titan-embed-image-v1"

    # ── Audio Transcription ────────────────────────────────────────────────────
    # Primary: AWS Transcribe (uses same AWS creds, no extra key needed).
    # Fallback: OpenAI Whisper via API (faster iteration during development).
    USE_AWS_TRANSCRIBE: bool = True
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="Required only when USE_AWS_TRANSCRIBE=False.",
    )
    WHISPER_MODEL: str = "whisper-1"

    # ── Video Processing ───────────────────────────────────────────────────────
    # Number of frames to uniformly sample from the full video duration.
    # Lower = faster processing, higher = better search quality.
    # ~15 frames is a good balance for most videos.
    SPLIT_FRAMES_COUNT: int = 15

    # Audio chunking — overlap avoids cutting sentences at boundaries.
    AUDIO_CHUNK_LENGTH: int = 10          # seconds per chunk
    AUDIO_OVERLAP_SECONDS: int = 1
    AUDIO_MIN_CHUNK_DURATION_SECONDS: int = 1

    # Frames are resized before embedding/captioning to reduce API cost.
    IMAGE_RESIZE_WIDTH: int = 1024
    IMAGE_RESIZE_HEIGHT: int = 768

    # Window (in seconds) around a matched frame used to build the output clip.
    DELTA_SECONDS_FRAME_INTERVAL: float = 5.0

    # Nova Pro scene-captioning: batch N consecutive frames in a single call
    # instead of one API call per frame (reduces latency & cost significantly).
    SCENE_CAPTION_BATCH_SIZE: int = 5
    CAPTION_MODEL_PROMPT: str = (
        "You are analyzing frames from a video. "
        "Describe concisely what is happening: key actions, objects, people, "
        "and any visible text. Be factual and specific."
    )

    # ── Search / Retrieval ─────────────────────────────────────────────────────
    # Top-k results returned per modality before similarity-based fusion.
    SPEECH_SEARCH_TOP_K: int = 3
    CAPTION_SEARCH_TOP_K: int = 3
    IMAGE_SEARCH_TOP_K: int = 3

    # Top-k context chunks passed to Nova Pro for answer generation.
    QA_CONTEXT_TOP_K: int = 5

    # Minimum similarity threshold — results below this are discarded.
    SIMILARITY_THRESHOLD: float = 0.25

    # ── Agent / Memory ─────────────────────────────────────────────────────────
    # How many past conversation turns to include in each LLM call.
    AGENT_MEMORY_SIZE: int = 20

    # Nova Lite is used for fast routing (does the query need a tool?).
    # Nova Pro is used for tool-selection, RAG answering, and general chat.
    # Typed Optional[str] so Pydantic accepts the None default; the
    # _resolve_model_defaults validator below fills them in at startup.
    ROUTING_MODEL: Optional[str] = Field(default=None)   # resolved in validator below
    TOOL_USE_MODEL: Optional[str] = Field(default=None)
    GENERAL_MODEL: Optional[str] = Field(default=None)

    # ── MCP Server ─────────────────────────────────────────────────────────────
    MCP_HOST: str = "0.0.0.0"
    MCP_PORT: int = 9090
    MCP_SERVER_URL: str = "http://localhost:9090/mcp"

    # ── API Server ─────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080

    # ── Observability (Opik / Comet ML) ───────────────────────────────────────
    OPIK_API_KEY: Optional[str] = None
    OPIK_WORKSPACE: str = "default"
    OPIK_PROJECT: str = "video-rag"

    # ── Registry / Cache ───────────────────────────────────────────────────────
    # Directory where processed-video metadata is persisted between restarts.
    REGISTRY_DIR: str = ".records"

    # ── Shared Media ───────────────────────────────────────────────────────────
    # Directory where uploaded videos and generated clips are stored.
    # Both the MCP process and the API process must have access to this path.
    SHARED_MEDIA_DIR: str = "shared_media"

    # ──────────────────────────────────────────────────────────────────────────
    @model_validator(mode="after")
    def _normalize_sonic_settings(self) -> "Settings":
        """Normalize Sonic settings loaded from .env (strip inline comments/whitespace)."""
        if self.SONIC_VOICE_ID:
            # Handle values like: "tiffany  # options: ..."
            cleaned = self.SONIC_VOICE_ID.split("#", 1)[0].strip().lower()
            self.SONIC_VOICE_ID = cleaned

        if not self.SONIC_VOICE_ID:
            self.SONIC_VOICE_ID = "tiffany"
        return self

    @model_validator(mode="after")
    def _resolve_model_defaults(self) -> "Settings":
        """
        Fill model-ID shortcuts with the concrete Bedrock model IDs if the
        caller has not overridden them explicitly in the environment.
        """
        if not self.ROUTING_MODEL:
            self.ROUTING_MODEL = self.NOVA_LITE_MODEL_ID
        if not self.TOOL_USE_MODEL:
            self.TOOL_USE_MODEL = self.NOVA_PRO_MODEL_ID
        if not self.GENERAL_MODEL:
            self.GENERAL_MODEL = self.NOVA_PRO_MODEL_ID
        return self

    @model_validator(mode="after")
    def _validate_transcription(self) -> "Settings":
        """Ensure OpenAI key is present when Whisper fallback is selected."""
        if not self.USE_AWS_TRANSCRIBE and not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY must be set when USE_AWS_TRANSCRIBE=False."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()