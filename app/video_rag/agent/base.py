"""
BaseAgent — abstract foundation for all agent implementations.

Handles:
- MCP client lifecycle (connect / disconnect)
- Tool discovery and filtering
- Prompt fetching from the MCP server
- Memory management
- Abstract `chat()` interface

Concrete subclasses (e.g. NovaAgent) implement `chat()` and any
provider-specific logic.
"""

import time
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from fastmcp import Client

from video_rag.agent.memory import Memory
from video_rag.agent.models import AssistantMessageResponse
from config import get_settings
from logging_config import get_logger, log_performance

settings = get_settings()
logger = get_logger("BaseAgent")


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        mcp_server_url: str,
        disable_tools: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.mcp_server_url = mcp_server_url
        self.disable_tools: List[str] = disable_tools or []

        # Each agent instance gets its own session memory.
        self._session_id = uuid.uuid4().hex[:8]
        self.memory = Memory(session_id=self._session_id)

        # Populated lazily by setup().
        self.tools: Optional[List] = None
        self.routing_prompt: Optional[str] = None
        self.tool_use_prompt: Optional[str] = None
        self.general_prompt: Optional[str] = None

        self._ready = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """
        Initialise async components.

        Must be awaited before the first call to `chat()`.
        Idempotent — safe to call on every request.
        """
        if self._ready:
            return

        setup_start = time.perf_counter()
        logger.info(
            "Initializing agent",
            name=self.name,
            session_id=self._session_id,
            mcp_server_url=self.mcp_server_url,
        )

        async with Client(self.mcp_server_url) as client:
            raw_tools = await client.list_tools()
            self.tools = self._filter_tools(raw_tools)
            
            logger.debug(
                "Tools discovered",
                total_tools=len(raw_tools),
                filtered_tools=len(self.tools),
                disabled=self.disable_tools,
                available=[t.name for t in self.tools],
            )

            self.routing_prompt  = await self._fetch_prompt(client, "routing_system_prompt")
            self.tool_use_prompt = await self._fetch_prompt(client, "tool_use_system_prompt")
            self.general_prompt  = await self._fetch_prompt(client, "general_system_prompt")
            
            logger.debug(
                "Prompts loaded",
                routing_prompt_len=len(self.routing_prompt) if self.routing_prompt else 0,
                tool_use_prompt_len=len(self.tool_use_prompt) if self.tool_use_prompt else 0,
                general_prompt_len=len(self.general_prompt) if self.general_prompt else 0,
            )

        setup_duration = (time.perf_counter() - setup_start) * 1000
        logger.info(
            "Agent ready",
            name=self.name,
            tools_count=len(self.tools),
            session_id=self._session_id,
            setup_duration_ms=setup_duration,
        )
        self._ready = True

    # ── MCP helpers ────────────────────────────────────────────────────────────

    async def call_tool(self, tool_name: str, args: dict) -> str:
        """Execute a single MCP tool call and return the text response."""
        call_start = time.perf_counter()
        
        logger.debug(
            "Calling MCP tool",
            tool_name=tool_name,
            args_keys=list(args.keys()),
            video_path=args.get("video_path"),
        )
        
        async with Client(self.mcp_server_url) as client:
            result = await client.call_tool(tool_name, args)
            # MCP SDK returns a CallToolResult; access .content for the list of blocks.
            response_text = ""
            if result.content:
                response_text = result.content[0].text
        
        call_duration = (time.perf_counter() - call_start) * 1000
        logger.info(
            "MCP tool call completed",
            tool_name=tool_name,
            response_length=len(response_text),
            duration_ms=call_duration,
        )
        
        return response_text

    async def _fetch_prompt(self, client: Client, prompt_name: str) -> str:
        try:
            mcp_prompt = await client.get_prompt(prompt_name)
            return mcp_prompt.messages[0].content.text
        except Exception as exc:
            logger.warning(f"Could not fetch prompt '{prompt_name}': {exc}")
            return ""

    def _filter_tools(self, tools: list) -> list:
        return [t for t in tools if t.name not in self.disable_tools]

    # ── Memory helpers ─────────────────────────────────────────────────────────

    def reset_memory(self) -> None:
        self.memory.reset()
        self._ready = False   # Force prompt re-fetch on next setup().

    def _build_history(self, system_prompt: str, user_message: str, n: int) -> List[dict]:
        """
        Construct the messages list for a Bedrock converse call.

        Layout:
            system   → system_prompt  (passed separately in Bedrock API)
            user     → oldest turn
            assistant→ …
            …
            user     → current message   ← always last
        """
        history = [
            {"role": r.role, "content": [{"text": r.content}]}
            for r in self.memory.get_latest(n)
        ]
        history.append({"role": "user", "content": [{"text": user_message}]})
        return history

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    async def chat(
        self,
        message: str,
        video_path: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> AssistantMessageResponse:
        """Process a user message and return the assistant response."""
        ...