"""
NovaAgent — concrete agent implementation using Amazon Nova via Bedrock.

Pipeline per user message
─────────────────────────
1. ROUTING  (Nova Lite, fast)
   Does this message need a video tool, or is it general conversation?
   → If no video_path is provided, always skip to general chat.

2. TOOL SELECTION  (Nova Pro, structured output)
   Which of the three tools should be called, and with what query?
   → Produces a ToolSelectionSchema JSON object.

3. TOOL EXECUTION  (MCP call)
   Calls the selected tool via the MCP server and captures the result.

4. RESPONSE GENERATION  (Nova Pro)
   Generates a natural-language response grounded in the tool output.
   Returns either VideoClipResponseSchema or QAResponseSchema depending
   on which tool was used.

5. MEMORY
   The user message + final assistant message are stored in Pixeltable.

Bedrock Converse API
────────────────────
We use `bedrock_runtime.converse()` (not `invoke_model`) throughout because
it provides a unified interface across all Nova model sizes, handles the
message format automatically, and supports structured JSON output via the
`toolConfig` / `toolChoice` mechanism without requiring a separate
`instructor` library.

For structured outputs we pass a single "tool" definition whose input schema
matches our Pydantic model, then force `toolChoice={"tool": {"name": …}}`.
The model is guaranteed to call that tool, giving us the fields we need.
"""

import json
import time
from typing import Any, Dict, List, Optional

import boto3
import opik
from opik import opik_context

from video_rag.agent.base import BaseAgent
from video_rag.agent.models import (
    AssistantMessageResponse,
    GeneralResponseSchema,
    QAResponseSchema,
    RoutingSchema,
    ToolSelectionSchema,
    VideoClipResponseSchema,
)
from config import get_settings
from logging_config import get_logger, log_performance, LoggingContext

settings = get_settings()
logger = get_logger("NovaAgent")


# ── Bedrock client (lazy singleton) ───────────────────────────────────────────

_bedrock = None


def _get_bedrock():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
    return _bedrock


# ── Structured-output helper ───────────────────────────────────────────────────

def _pydantic_to_bedrock_tool(schema_cls) -> Dict[str, Any]:
    """
    Convert a Pydantic model class into a Bedrock tool definition.

    We use this to force structured JSON output: we define one tool whose
    inputSchema matches our model, then set toolChoice to force Nova to
    call it.  The "tool call" arguments are our structured response.
    """
    name = schema_cls.__name__
    json_schema = schema_cls.model_json_schema()
    return {
        "toolSpec": {
            "name": name,
            "description": f"Return a structured {name} response.",
            "inputSchema": {"json": json_schema},
        }
    }


def _converse_structured(
    model_id: str,
    system_prompt: str,
    messages: List[Dict],
    schema_cls,
    temperature: float = 0.1,
) -> Any:
    """
    Call Bedrock Converse with forced structured output.

    Returns a parsed instance of *schema_cls*.
    """
    tool_def = _pydantic_to_bedrock_tool(schema_cls)
    tool_name = schema_cls.__name__

    response = _get_bedrock().converse(
        modelId=model_id,
        system=[{"text": system_prompt}],
        messages=messages,
        toolConfig={
            "tools": [tool_def],
            "toolChoice": {"tool": {"name": tool_name}},
        },
        inferenceConfig={"temperature": temperature, "maxTokens": 1024},
    )

    # Extract the tool-use block from the response.
    content_blocks = response["output"]["message"]["content"]
    for block in content_blocks:
        if block.get("toolUse", {}).get("name") == tool_name:
            raw = block["toolUse"]["input"]
            return schema_cls.model_validate(raw)

    raise ValueError(f"Nova did not return a {tool_name} tool call.")


def _converse_text(
    model_id: str,
    system_prompt: str,
    messages: List[Dict],
    temperature: float = 0.5,
    max_tokens: int = 1024,
) -> str:
    """Call Bedrock Converse for plain text output. Returns the response string."""
    response = _get_bedrock().converse(
        modelId=model_id,
        system=[{"text": system_prompt}],
        messages=messages,
        inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
    )
    return response["output"]["message"]["content"][0]["text"]


# ── NovaAgent ──────────────────────────────────────────────────────────────────

class NovaAgent(BaseAgent):
    """
    Agent implementation powered by Amazon Nova Lite (routing) and
    Nova Pro (tool selection, response generation, general chat).
    """

    def __init__(
        self,
        mcp_server_url: str,
        disable_tools: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            name="VideoRAG",
            mcp_server_url=mcp_server_url,
            disable_tools=disable_tools or ["process_video"],
        )

    # ── Public entry point ─────────────────────────────────────────────────────

    @opik.track(name="agent.chat")
    async def chat(
        self,
        message: str,
        video_path: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> AssistantMessageResponse:
        """
        Process one user turn and return the assistant response.

        Args:
            message:      The user's text message.
            video_path:   Path to the video currently loaded in the UI (optional).
            image_base64: Base64 query image supplied by the user (optional).
        """
        start_time = time.perf_counter()
        
        logger.info(
            "Chat request received",
            message_preview=message[:100] + "..." if len(message) > 100 else message,
            video_path=video_path,
            has_image=bool(image_base64),
            session_id=self._session_id,
        )
        
        await self.setup()

        # ── Step 1: Routing ────────────────────────────────────────────────────
        route_start = time.perf_counter()
        use_tool = video_path and self._route(message)
        route_duration = (time.perf_counter() - route_start) * 1000
        
        logger.info(
            "Routing decision completed",
            use_tool=use_tool,
            duration_ms=route_duration,
            model_id=settings.ROUTING_MODEL,
        )

        # ── Step 2–4: Tool path ────────────────────────────────────────────────
        if use_tool:
            response = await self._run_tool_pipeline(message, video_path, image_base64)
        else:
            response = self._run_general(message)

        # ── Step 5: Memory ─────────────────────────────────────────────────────
        self.memory.add_pair(message, response.message)
        
        total_duration = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Chat request completed",
            total_duration_ms=total_duration,
            used_tool=use_tool,
            response_length=len(response.message),
            has_clip=bool(response.clip_path),
            session_id=self._session_id,
        )

        return response

    # ── Step 1: Routing ────────────────────────────────────────────────────────

    # Keywords that ALWAYS trigger tool use when a video is loaded
    _VIDEO_KEYWORDS = {
        "summarize", "summarise", "summary", "describe", "about", 
        "what is", "what's", "tell me", "explain", "show me", "find",
        "search", "look for", "clip", "scene", "moment", "watch",
        "who", "what", "where", "when", "how", "why",
    }

    def _keyword_match(self, message: str) -> bool:
        """Check if message contains video-related keywords."""
        msg_lower = message.lower()
        return any(kw in msg_lower for kw in self._VIDEO_KEYWORDS)

    @opik.track(name="agent.route")
    def _route(self, message: str) -> bool:
        """
        Use Nova Lite to decide if a tool call is needed.

        Nova Lite is fast and cheap — ideal for this binary classification.
        Falls back to keyword matching if LLM returns False (safety net).
        """
        # Fast keyword check as fallback
        keyword_match = self._keyword_match(message)
        
        messages = [{"role": "user", "content": [{"text": message}]}]
        try:
            result = _converse_structured(
                model_id=settings.ROUTING_MODEL,
                system_prompt=self.routing_prompt,
                messages=messages,
                schema_cls=RoutingSchema,
                temperature=0.0,
            )
            llm_decision = result.tool_use
            
            # Use tool if EITHER LLM says yes OR keywords match (safety net)
            final_decision = llm_decision or keyword_match
            
            logger.info(
                f"Route decision: {'✅ YES' if final_decision else '❌ NO'} "
                f"(LLM={llm_decision}, keywords={keyword_match})",
                message_preview=message[:50] + "..." if len(message) > 50 else message,
            )
            return final_decision
        except Exception as exc:
            logger.warning(f"Routing failed ({exc}), using keyword fallback={keyword_match}.")
            return keyword_match

    # ── Steps 2–4: Tool pipeline ───────────────────────────────────────────────

    @opik.track(name="agent.tool_pipeline")
    async def _run_tool_pipeline(
        self,
        message: str,
        video_path: str,
        image_base64: Optional[str],
    ) -> AssistantMessageResponse:
        """Select a tool → execute it → generate a grounded response."""
        pipeline_start = time.perf_counter()

        # Step 2: Tool selection.
        select_start = time.perf_counter()
        selection = self._select_tool(message, image_provided=bool(image_base64))
        select_duration = (time.perf_counter() - select_start) * 1000
        
        logger.info(
            "Tool selected",
            tool_name=selection.tool_name,
            user_query=selection.user_query[:80] + "..." if selection.user_query and len(selection.user_query) > 80 else selection.user_query,
            duration_ms=select_duration,
            model_id=settings.TOOL_USE_MODEL,
        )

        # Step 3: Tool execution.
        exec_start = time.perf_counter()
        tool_result = await self._execute_tool(
            selection=selection,
            video_path=video_path,
            image_base64=image_base64,
            original_message=message,
        )
        exec_duration = (time.perf_counter() - exec_start) * 1000
        
        logger.info(
            "Tool executed",
            tool_name=selection.tool_name,
            result_preview=tool_result[:150] + "..." if len(tool_result) > 150 else tool_result,
            result_length=len(tool_result),
            duration_ms=exec_duration,
        )

        # Step 4: Response generation.
        gen_start = time.perf_counter()
        response = self._generate_tool_response(
            tool_name=selection.tool_name,
            tool_result=tool_result,
            original_message=message,
        )
        gen_duration = (time.perf_counter() - gen_start) * 1000
        
        pipeline_duration = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            "Tool pipeline completed",
            tool_name=selection.tool_name,
            select_ms=select_duration,
            execute_ms=exec_duration,
            generate_ms=gen_duration,
            total_ms=pipeline_duration,
        )
        
        return response

    @opik.track(name="agent.select_tool")
    def _select_tool(self, message: str, image_provided: bool) -> ToolSelectionSchema:
        """Use Nova Pro to pick the right tool and extract the query."""
        system_prompt = self.tool_use_prompt.format(
            is_image_provided=image_provided
        )
        messages = self._build_history(system_prompt, message, settings.AGENT_MEMORY_SIZE)
        return _converse_structured(
            model_id=settings.TOOL_USE_MODEL,
            system_prompt=system_prompt,
            messages=messages,
            schema_cls=ToolSelectionSchema,
            temperature=0.0,
        )

    @opik.track(name="agent.execute_tool")
    async def _execute_tool(
        self,
        selection: ToolSelectionSchema,
        video_path: str,
        image_base64: Optional[str],
        original_message: str,
    ) -> str:
        """Dispatch the MCP tool call."""
        args: Dict[str, Any] = {"video_path": video_path}
        
        tool_name = selection.tool_name
        query = selection.user_query or original_message

        if tool_name == "get_video_clip_by_time":
            if selection.start_seconds is not None:
                args["start_seconds"] = selection.start_seconds
                args["end_seconds"] = selection.end_seconds or (selection.start_seconds + 5) # Default duration?
            else:
                logger.warning("Time tool selected but no start_seconds provided - falling back to query search.")
                tool_name = "get_video_clip_from_query"
                args["user_query"] = query

        elif tool_name == "get_video_clip_from_image":
            if not image_base64:
                logger.warning("Image tool selected but no image provided — falling back to query tool.")
                tool_name = "get_video_clip_from_query"
                args["user_query"] = query
            else:
                args["user_image"] = image_base64
        else:
            args["user_query"] = query

        try:
            return await self.call_tool(tool_name, args)
        except Exception as exc:
            logger.error(f"Tool '{tool_name}' failed: {exc}")
            return f"Error: {exc}"

    @opik.track(name="agent.generate_response")
    def _generate_tool_response(
        self,
        tool_name: str,
        tool_result: str,
        original_message: str,
    ) -> AssistantMessageResponse:
        """
        Generate a natural-language response grounded in the tool result.

        For clip tools the result is a file path; for QA it is already prose.
        Nova Pro wraps the result in a personality-appropriate message.
        """
        is_clip_tool = tool_name in (
            "get_video_clip_from_query",
            "get_video_clip_from_image",
            "get_video_clip_by_time",
        )
        schema_cls = VideoClipResponseSchema if is_clip_tool else QAResponseSchema

        context = (
            f"The tool returned this clip path: {tool_result}"
            if is_clip_tool
            else f"The tool returned this answer: {tool_result}"
        )
        system_prompt = (
            f"{self.general_prompt}\n\n"
            f"Tool context:\n{context}\n\n"
            "Use the tool context to craft your response to the user. "
            "Be concise and engaging."
        )
        messages = [{"role": "user", "content": [{"text": original_message}]}]

        try:
            result = _converse_structured(
                model_id=settings.GENERAL_MODEL,
                system_prompt=system_prompt,
                messages=messages,
                schema_cls=schema_cls,
            )
            clip_path = result.clip_path if is_clip_tool else None
            return AssistantMessageResponse(message=result.message, clip_path=clip_path)
        except Exception as exc:
            logger.error(f"Response generation failed: {exc}")
            # Graceful degradation — return the raw tool output.
            return AssistantMessageResponse(
                message=tool_result,
                clip_path=tool_result if is_clip_tool else None,
            )

    # ── General conversation ───────────────────────────────────────────────────

    @opik.track(name="agent.general_chat")
    def _run_general(self, message: str) -> AssistantMessageResponse:
        """Handle messages that don't require a tool (general conversation)."""
        messages = self._build_history(
            self.general_prompt, message, settings.AGENT_MEMORY_SIZE
        )
        try:
            result = _converse_structured(
                model_id=settings.GENERAL_MODEL,
                system_prompt=self.general_prompt,
                messages=messages,
                schema_cls=GeneralResponseSchema,
                temperature=0.7,
            )
            return AssistantMessageResponse(message=result.message)
        except Exception as exc:
            logger.error(f"General chat failed: {exc}")
            # Last-resort fallback to unstructured text.
            text = _converse_text(
                model_id=settings.GENERAL_MODEL,
                system_prompt=self.general_prompt,
                messages=messages,
            )
            return AssistantMessageResponse(message=text)