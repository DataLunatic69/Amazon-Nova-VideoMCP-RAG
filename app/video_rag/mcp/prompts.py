"""
System prompts for the three LLM stages in the agent pipeline.

Prompts are versioned in Opik (Comet ML) when credentials are available.
If Opik is unreachable the hardcoded fallbacks are used transparently.
"""

from loguru import logger

logger = logger.bind(name="Prompts")

# ── Hardcoded fallbacks ────────────────────────────────────────────────────────

_ROUTING_PROMPT = """
You are a routing assistant. Given a conversation history, determine whether
the user's latest message requires operating on a video — specifically:

- Retrieving a clip from a specific moment or scene
- Answering a factual question about the video's content
- Finding a clip that matches a provided image

Output a single boolean: true if a tool should be used, false otherwise.
""".strip()

_TOOL_USE_PROMPT = """
You are a video assistant. Select the correct tool based on the user's request.

Available tools:
- get_video_clip_from_query  : find a clip using a text description
- get_video_clip_from_image  : find a clip visually similar to an image
- ask_question_about_video   : answer a factual question about the video

Rules:
- If the user provided an image, ALWAYS use get_video_clip_from_image.
- If the user wants to *watch* something, use get_video_clip_from_query.
- If the user wants to *know* something, use ask_question_about_video.

Context:
- Image provided: {is_image_provided}
""".strip()

_GENERAL_PROMPT = """
You are a knowledgeable and enthusiastic video assistant. You help users
explore, search, and understand video content.

You have deep knowledge of film, cinematography, and video production.
Keep your responses concise, engaging, and grounded in what the video
actually contains. When you don't know something, say so clearly.
""".strip()


# ── Opik-backed prompt helpers ─────────────────────────────────────────────────

def _fetch_or_create(client, prompt_id: str, fallback: str) -> str:
    """
    Try to load *prompt_id* from Opik; create it from *fallback* if absent.
    Returns the prompt text, falling back to *fallback* on any error.
    """
    try:
        prompt = client.get_prompt(prompt_id)
        if prompt is None:
            prompt = client.create_prompt(name=prompt_id, prompt=fallback)
            logger.info(f"Created new Opik prompt '{prompt_id}'.")
        return prompt.prompt
    except Exception as e:
        logger.warning(f"Opik unavailable for '{prompt_id}' ({e}). Using hardcoded fallback.")
        return fallback


def _get_opik_client():
    try:
        import opik
        return opik.Opik()
    except Exception:
        return None


# ── Public accessors ───────────────────────────────────────────────────────────

def routing_system_prompt() -> str:
    client = _get_opik_client()
    if client:
        return _fetch_or_create(client, "routing-system-prompt", _ROUTING_PROMPT)
    return _ROUTING_PROMPT


def tool_use_system_prompt() -> str:
    client = _get_opik_client()
    if client:
        return _fetch_or_create(client, "tool-use-system-prompt", _TOOL_USE_PROMPT)
    return _TOOL_USE_PROMPT


def general_system_prompt() -> str:
    client = _get_opik_client()
    if client:
        return _fetch_or_create(client, "general-system-prompt", _GENERAL_PROMPT)
    return _GENERAL_PROMPT