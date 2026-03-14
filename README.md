# Video RAG — Multimodal Video Intelligence on Amazon Bedrock

A production-ready multimodal video search and question-answering system built on **Amazon Nova** (Lite, Pro, Sonic, and Titan Multimodal Embeddings) via Amazon Bedrock. Users can upload videos, ask natural-language questions, retrieve precise clips by text or image query, and interact through a real-time voice interface powered by Nova Sonic.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Agent Pipeline](#agent-pipeline)
- [Video Ingestion Pipeline](#video-ingestion-pipeline)
- [Search and Retrieval](#search-and-retrieval)

---

## Architecture Overview

The system is split into two long-running processes:

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                            │
│              Next.js  ·  http://localhost:3000              │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────┐
│                      FastAPI Server                         │
│              REST API  ·  http://localhost:8080             │
│   /api/v1/chat  ·  /api/v1/video  ·  /api/v1/video/media   │
└───────────────────────────┬─────────────────────────────────┘
                            │ FastMCP (HTTP)
┌───────────────────────────▼─────────────────────────────────┐
│                       MCP Tool Server                       │
│              FastMCP  ·  http://localhost:9090/mcp          │
│   process_video  ·  get_video_clip_from_query               │
│   get_video_clip_from_image  ·  ask_question_about_video    │
│   get_video_clip_by_time                                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    Amazon Bedrock                           │
│   Nova Lite (routing)  ·  Nova Pro (tool use, RAG, chat)    │
│   Nova Sonic (voice)   ·  Titan Multimodal Embeddings       │
└─────────────────────────────────────────────────────────────┘
```

The **FastAPI server** hosts the NovaAgent, which orchestrates a four-stage pipeline (route → select tool → execute → respond). The **MCP tool server** owns all video processing and retrieval logic and exposes it as typed MCP tools. Pixeltable serves as the unified vector store and media table backend.

---

## Key Features

**Multimodal Search**
- Text-to-clip search across transcribed speech and AI-generated scene captions
- Image-to-clip search using Amazon Titan Multimodal Embeddings (unified vector space — no separate CLIP model required)
- Time-based clip extraction for precise timestamp queries

**Intelligent Agent**
- Two-stage routing: Nova Lite classifies the intent, Nova Pro selects and executes the tool
- Structured JSON output (Pydantic-validated) at every LLM step — no ad-hoc string parsing
- Sliding-window conversation memory backed by Pixeltable

**Video Ingestion Pipeline**
- Frame extraction (PyAV) with configurable sampling rate
- Audio transcription via AWS Transcribe (primary) or OpenAI Whisper (fallback)
- Batched scene captioning with Nova Pro — multiple consecutive frames per API call for efficiency
- All embeddings produced by Titan Multimodal Embeddings for a single unified vector space

**Voice Interface**
- Real-time bidirectional speech via Nova Sonic (WebSocket streaming)
- Configurable voice ID (Tiffany, Matthew, Amy)

**Observability**
- Full LLM trace capture with Opik (Comet ML) — every routing, tool-selection, and generation call is tracked
- Structured logging via Loguru with per-component context

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language Models | Amazon Nova Lite, Nova Pro, Nova Sonic |
| Embeddings | Amazon Titan Multimodal Embeddings V1 |
| Vector Store + Media Tables | Pixeltable |
| API Framework | FastAPI + Uvicorn |
| Tool Server | FastMCP |
| Video Processing | PyAV, MoviePy, OpenCV, ffmpeg |
| Audio Transcription | AWS Transcribe / OpenAI Whisper |
| Observability | Opik (Comet ML) |
| Configuration | Pydantic Settings |

---

## Project Structure

```
video-rag/
├── app/
│   ├── config.py                        # Centralised Pydantic settings
│   ├── logging_config.py
│   ├── api/
│   │   ├── app.py                       # FastAPI application factory
│   │   ├── routers.py
│   │   ├── schemas.py
│   │   └── endpoints/v1/
│   │       ├── chat.py                  # POST /chat, DELETE /chat/memory
│   │       └── video.py                 # Upload, process, status, media
│   └── video_rag/
│       ├── agent/
│       │   ├── base.py                  # Abstract agent (MCP lifecycle, memory)
│       │   ├── nova_agent.py            # Concrete Nova implementation
│       │   ├── memory.py                # Pixeltable-backed conversation memory
│       │   └── models.py                # Pydantic schemas for all LLM I/O
│       └── mcp/
│           ├── server.py                # FastMCP server registration
│           ├── tools.py                 # MCP tool implementations
│           ├── prompts.py               # System prompts (Opik-versioned)
│           ├── resources.py             # MCP resource: list_videos
│           └── video/
│               ├── processor.py         # Full ingestion pipeline
│               ├── search_engine.py     # Similarity search (speech, caption, image)
│               ├── nova_client.py       # All Bedrock API calls
│               ├── transcription.py     # AWS Transcribe + Whisper
│               ├── media.py             # ffmpeg clip extraction
│               ├── registry.py          # On-disk JSON index registry
│               ├── models.py            # CachedTable, CachedTableMetadata
│               └── functions.py         # Pixeltable UDFs
├── main.py                              # CLI entry point (click)
├── gunicorn_config.py                   # Production deployment config
├── run.sh                               # Development start script
├── pyproject.toml
└── .env.example
```

---

## Prerequisites

- Python 3.12 or later
- Node.js 18+ (for the frontend)
- ffmpeg installed and available in `PATH`
- An AWS account with Amazon Bedrock access enabled for the following models:
  - `amazon.nova-lite-v1:0`
  - `amazon.nova-pro-v1:0`
  - `amazon.nova-sonic-v1:0`
  - `amazon.titan-embed-image-v1`
- (Optional) An S3 bucket for AWS Transcribe, or an OpenAI API key for Whisper fallback

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd video-rag

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -e .

# Copy and edit the environment file
cp .env.example .env
```

---

## Configuration

All settings are managed via `.env` at the project root. The full list of available variables is documented in `.env.example`. The minimum required values are:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# Required only when USE_AWS_TRANSCRIBE=true
AWS_S3_TRANSCRIBE_BUCKET=your-bucket-name

# Required only when USE_AWS_TRANSCRIBE=false
OPENAI_API_KEY=sk-...
```

Key tuning parameters (all have sensible defaults):

| Variable | Default | Description |
|---|---|---|
| `SPLIT_FRAMES_COUNT` | 15 | Frames sampled per video |
| `SCENE_CAPTION_BATCH_SIZE` | 5 | Frames per Nova Pro captioning call |
| `SIMILARITY_THRESHOLD` | 0.25 | Minimum similarity to return a result |
| `QA_CONTEXT_TOP_K` | 5 | Context chunks passed to RAG |
| `DELTA_SECONDS_FRAME_INTERVAL` | 5.0 | Clip window around matched frame (seconds) |

---

## Running the Application

### Development (recommended)

The `run.sh` script starts both servers, writes logs to `logs/`, and tails them in your terminal.

```bash
# Start both MCP server and API server
./run.sh

# Start only one server
./run.sh api
./run.sh mcp

# Stop background servers
./run.sh stop
```

### Manual start

```bash
# Terminal 1 — MCP tool server (port 9090)
python main.py mcp

# Terminal 2 — FastAPI server (port 8080)
python main.py api
```

### Production (Gunicorn)

```bash
cd app
gunicorn -c ../gunicorn_config.py api.app:app
```

Service endpoints once running:

| Service | URL |
|---|---|
| API server | http://localhost:8080 |
| Interactive API docs | http://localhost:8080/docs |
| MCP tool server | http://localhost:9090/mcp |
| Frontend | http://localhost:3000 |

---

## API Reference

### Video Workflow

```
POST   /api/v1/video/upload              Upload a video file
POST   /api/v1/video/process             Start background ingestion; returns task_id
GET    /api/v1/video/task-status/{id}    Poll ingestion status
GET    /api/v1/video/media/{file_path}   Serve a generated clip or uploaded video
```

### Chat

```
POST   /api/v1/chat                      Send a message to the agent
DELETE /api/v1/chat/memory               Clear conversation history
GET    /api/v1/chat/history              Retrieve recent messages
```

**Chat request body:**

```json
{
  "message": "What happens in the opening scene?",
  "video_path": "shared_media/my_video.mp4",
  "image_base64": null
}
```

**Chat response:**

```json
{
  "message": "The opening scene shows a wide aerial shot of the city at dawn...",
  "clip_path": "shared_media/clip_a1b2c3d4.mp4"
}
```

The `clip_path` field is non-null only when a clip was extracted. Serve it via `GET /api/v1/video/media/{clip_path}`.

---

## Agent Pipeline

Each user message passes through four stages:

```
User message
    │
    ▼
1. ROUTING  (Nova Lite)
   └── Is a video tool needed?  Yes / No
    │
    ▼ (if Yes)
2. TOOL SELECTION  (Nova Pro, structured JSON)
   └── Which tool?  get_video_clip_from_query
                    get_video_clip_from_image
                    get_video_clip_by_time
                    ask_question_about_video
    │
    ▼
3. TOOL EXECUTION  (MCP call)
   └── Returns clip path or RAG answer text
    │
    ▼
4. RESPONSE GENERATION  (Nova Pro, structured JSON)
   └── Wraps result in natural language

    ▼ (if No video tool needed)
    General conversation  (Nova Pro)
```

All structured outputs are Pydantic-validated (`RoutingSchema`, `ToolSelectionSchema`, `VideoClipResponseSchema`, `QAResponseSchema`). If LLM-based routing fails, the agent falls back to keyword matching as a safety net.

---

## Video Ingestion Pipeline

When `process_video` is called, the following steps execute sequentially:

1. **Frame extraction** — PyAV samples `SPLIT_FRAMES_COUNT` frames uniformly across the video duration. Frames are resized to `IMAGE_RESIZE_WIDTH x IMAGE_RESIZE_HEIGHT` before embedding.

2. **Audio extraction and chunking** — The audio track is split into overlapping `AUDIO_CHUNK_LENGTH`-second segments. Overlap prevents sentence boundaries from being cut.

3. **Transcription** — Each audio chunk is transcribed using AWS Transcribe (async job via S3) or OpenAI Whisper. The resulting text is stored as a Pixeltable column.

4. **Scene captioning** — Frames are grouped into batches of `SCENE_CAPTION_BATCH_SIZE` and sent to Nova Pro in a single multi-image API call. This significantly reduces both latency and cost compared to one call per frame.

5. **Embedding** — Text (transcripts and captions) and images (frames) are all embedded using **Amazon Titan Multimodal Embeddings V1**, producing a single unified vector space. This eliminates the need for separate CLIP and text-embedding models.

6. **Index registration** — Table names and metadata are persisted to a JSON snapshot in `.records/` so the server does not need to re-index on restart.

---

## Search and Retrieval

`VideoSearchEngine` exposes three search methods, all backed by Titan Multimodal Embeddings cosine similarity:

| Method | Index searched | Use case |
|---|---|---|
| `search_by_speech(query)` | Audio transcript chunks | Dialogue, narration, spoken keywords |
| `search_by_caption(query)` | Nova Pro scene captions | Visual descriptions, on-screen actions |
| `search_by_image(image_b64)` | Extracted video frames | Visual similarity to a reference image |

For clip retrieval (`get_video_clip_from_query`), both speech and caption indexes are queried and the result with the higher similarity score wins. Results below `SIMILARITY_THRESHOLD` are discarded.

For RAG QA (`ask_question_about_video`), results from both speech and caption indexes are merged, deduplicated, and re-ranked by similarity before being passed as context to Nova Pro.