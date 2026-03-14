import click
from fastmcp import FastMCP
from fastmcp.prompts import Prompt
from fastmcp.resources import FunctionResource
from fastmcp.tools import Tool

from config import get_settings
from logging_config import setup_logging, get_logger, log_startup_banner
from video_rag.mcp.prompts import general_system_prompt, routing_system_prompt, tool_use_system_prompt
from video_rag.mcp.resources import list_videos
from video_rag.mcp.tools import (
    ask_question_about_video,
    get_video_clip_by_time,
    get_video_clip_from_image,
    get_video_clip_from_query,
    process_video,
)

# Initialize logging
setup_logging()
logger = get_logger("MCPServer")

settings = get_settings()

mcp = FastMCP("VideoRAG")


# ── Prompts ────────────────────────────────────────────────────────────────────

mcp.add_prompt(Prompt.from_function(
    routing_system_prompt,
    name="routing_system_prompt",
    description="Routing prompt — decides if a user query needs a tool call.",
    tags={"prompt", "routing"},
))

mcp.add_prompt(Prompt.from_function(
    tool_use_system_prompt,
    name="tool_use_system_prompt",
    description="Tool-use prompt — selects the right tool for the query.",
    tags={"prompt", "tool_use"},
))

mcp.add_prompt(Prompt.from_function(
    general_system_prompt,
    name="general_system_prompt",
    description="General conversational prompt.",
    tags={"prompt", "general"},
))


# ── Resources ──────────────────────────────────────────────────────────────────

mcp.add_resource(FunctionResource(
    fn=list_videos,
    uri="file:///app/.records/records.json",
    name="list_videos",
    description="List all videos that have been processed and indexed.",
    tags={"resource"},
))


# ── Tools ──────────────────────────────────────────────────────────────────────

mcp.add_tool(Tool.from_function(
    process_video,
    name="process_video",
    description=(
        "Ingest a video file: extract frames, transcribe audio, generate "
        "scene captions, and build multimodal search indexes. "
        "Must be called before any search or QA tool."
    ),
    tags={"video", "ingest"},
))

mcp.add_tool(Tool.from_function(
    get_video_clip_from_query,
    name="get_video_clip_from_query",
    description=(
        "Find the most relevant clip in the video for a natural-language query. "
        "Searches across transcribed speech AND frame captions, then returns "
        "the path to the trimmed clip."
    ),
    tags={"video", "clip", "search"},
))

mcp.add_tool(Tool.from_function(
    get_video_clip_by_time,
    name="get_video_clip_by_time",
    description=(
        "Extract a clip from the video at a specific time range. "
        "Use when the user asks for 'the first N seconds', 'from X to Y seconds', "
        "'the last N seconds', or any other time-based clip request. "
        "Provide start_seconds and end_seconds as floats."
    ),
    tags={"video", "clip", "time"},
))

mcp.add_tool(Tool.from_function(
    get_video_clip_from_image,
    name="get_video_clip_from_image",
    description=(
        "Find the video clip that is visually most similar to a provided image. "
        "The image must be supplied as a base64-encoded JPEG/PNG string."
    ),
    tags={"video", "clip", "image"},
))

mcp.add_tool(Tool.from_function(
    ask_question_about_video,
    name="ask_question_about_video",
    description=(
        "Answer a factual question about the video using RAG: retrieve the "
        "most relevant transcript chunks and captions, then generate a grounded "
        "answer with Amazon Nova Pro. Use this when the user wants information "
        "rather than a clip."
    ),
    tags={"video", "qa", "rag"},
))


# ── Entrypoint ─────────────────────────────────────────────────────────────────

@click.command()
@click.option("--host", default=settings.MCP_HOST, show_default=True)
@click.option("--port", default=settings.MCP_PORT, show_default=True)
@click.option("--transport", default="streamable-http", show_default=True)
def run_mcp(host: str, port: int, transport: str) -> None:
    """Start the FastMCP server."""
    log_startup_banner(
        app_name="Video RAG MCP Server",
        version="0.1.0",
        endpoints={
            "MCP": f"http://{host}:{port}/mcp",
            "Transport": transport,
        },
    )
    
    logger.info(
        "Starting MCP server",
        host=host,
        port=port,
        transport=transport,
        tools_count=5,
        prompts_count=3,
    )
    
    mcp.run(host=host, port=port, transport=transport)


if __name__ == "__main__":
    run_mcp()