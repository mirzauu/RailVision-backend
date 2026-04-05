import json
import logging
import os
import re
import sys
from typing import AsyncGenerator, Dict, List, Optional

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage
from claude_agent_sdk.types import StreamEvent

from src.infrastructure.agents.reasoning_manager import get_reasoning_manager
from src.domain.agents.base import (
    AgentConfig,
    ChatAgent,
    ChatAgentResponse,
    ChatContext,
    TaskConfig,
    ToolCallEventType,
    ToolCallResponse,
)

logger = logging.getLogger(__name__)


class   ClaudeChatAgent(ChatAgent):
    """Agent implementation using the Claude Agent SDK (claude-agent-sdk).

    This wraps ``claude_agent_sdk.query()`` to stream events from Claude Code
    and maps them into the same ``ChatAgentResponse`` / ``ToolCallResponse``
    Pydantic schemas used by PydanticChatAgent so the output is identical.
    """

    def __init__(self, config: AgentConfig):
        self.tasks = config.tasks
        self.max_iter = config.max_iter

        # The Claude Agent SDK requires ANTHROPIC_API_KEY, but this project
        # stores it as CLAUDE_API_KEY. Bridge the gap if needed.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            claude_key = os.environ.get("CLAUDE_API_KEY")
            if claude_key:
                os.environ["ANTHROPIC_API_KEY"] = claude_key
                logger.info("Set ANTHROPIC_API_KEY from CLAUDE_API_KEY for Claude SDK")

        # Build the system prompt from agent config (matches pydantic_agent format)
        self.system_prompt = (
            f"Role: {config.role}\n"
            f"Goal: {config.goal}\n"
            f"Backstory: {config.backstory}. Respond to the user query"
        )

        # Default Claude Agent SDK options
        self.allowed_tools = ["Skill"]

    def _create_task_description(self, task_config: TaskConfig, ctx: ChatContext) -> str:
        """Build the full prompt — identical logic to PydanticChatAgent."""
        return (
            f"\n                CONTEXT:\n"
            f"                User Query: {ctx.query}\n"
            f"                \n"
            f"                Additional Context:\n"
            f"                {ctx.additional_context if ctx.additional_context != '' else 'no additional context'}\n\n"
            f"                TASK:\n"
            f"                {task_config.description}\n\n"
            f"                Expected Output:\n"
            f"                {task_config.expected_output}\n\n"
            f"                INSTRUCTIONS:\n"
            f"                1. Use the available tools to gather information\n"
            f"                2. Process and synthesize the gathered information\n"
            f"                3. Format your response in markdown, make sure it's well formatted\n"
            f"                4. Include relevant code snippets and file references\n"
            f"                5. Provide clear explanations\n"
            f"                6. Verify your output before submitting\n\n"
            f"                IMPORTANT:\n"
            f"                - Use tools efficiently and avoid unnecessary API calls\n"
            f"                - Only use the tools listed below\n\n"
            f"                With above information answer the user query: {ctx.query}\n"
            f"            "
        )

    def _build_history_prompt(self, history: List[Dict[str, str]]) -> str:
        """Flatten chat history into a text block for the SDK prompt.

        The Claude Agent SDK doesn't accept structured message history — we
        serialize it into a readable conversation block prepended to the prompt.
        """
        if not history:
            return ""

        lines = ["--- Previous Conversation History ---"]
        for msg in history:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content}")
        lines.append("--- End of History ---\n")
        return "\n".join(lines)

    def _build_options(self) -> ClaudeAgentOptions:
        """Construct the SDK options shared by run() and run_stream()."""
        return ClaudeAgentOptions(
            setting_sources=["user", "project"],
            allowed_tools=self.allowed_tools,
            include_partial_messages=True,
        )

    def _build_full_prompt(self, ctx: ChatContext) -> str:
        """Combine system prompt, history, and task into the full prompt."""
        task_desc = self._create_task_description(self.tasks[0], ctx)
        history_block = self._build_history_prompt(ctx.history)

        parts = [self.system_prompt, ""]
        if history_block:
            parts.append(history_block)
        parts.append(task_desc)
        return "\n".join(parts)

    @staticmethod
    def _extract_citations(text: str) -> List[str]:
        """Extract file references from the response text (same regex as reference)."""
        files = re.findall(
            r'([a-zA-Z0-9_\-\./]+\.(?:py|md|txt|json|yaml|yml|toml|cfg|ini|csv|sql|sh|js|ts|jsx|tsx|html|css))',
            text,
        )
        return sorted(set(files)) if files else []

    # ------------------------------------------------------------------
    # run() — non-streaming, collects everything then returns
    # ------------------------------------------------------------------
    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        logger.info("running claude-agent-sdk agent")
        prompt = self._build_full_prompt(ctx)
        options = self._build_options()

        agent_response = ChatAgentResponse(response="", tool_calls=[], citations=[])
        active_tool_calls: Dict[str, ToolCallResponse] = {}
        partial_json_buffer: Dict[str, str] = {}

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, StreamEvent):
                    self._process_stream_event(
                        message, agent_response, active_tool_calls, partial_json_buffer
                    )
                elif isinstance(message, ResultMessage):
                    self._process_result_message(message, agent_response)
                else:
                    self._process_other_message(message, agent_response)
        except Exception as e:
            logger.error(f"Claude SDK agent run() error: {e}")
            if not agent_response.response:
                agent_response.response = f"[Service Alert: Claude SDK error: {str(e)}]"

        # Extract citations from the final response
        agent_response.citations = self._extract_citations(agent_response.response)
        return agent_response

    # ------------------------------------------------------------------
    # run_stream() — yields ChatAgentResponse chunks as events arrive
    # ------------------------------------------------------------------
    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        logger.info("running claude-agent-sdk agent (streaming)")
        prompt = self._build_full_prompt(ctx)
        options = self._build_options()

        active_tool_calls: Dict[str, ToolCallResponse] = {}
        partial_json_buffer: Dict[str, str] = {}
        full_response = ""
        reasoning_mgr = get_reasoning_manager()

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, StreamEvent):
                    event = message.event
                    etype = event.get("type")

                    # --- content_block_start ---
                    if etype == "content_block_start":
                        block = event.get("content_block", {})
                        if block.get("type") == "tool_use":
                            call_id = block.get("id", "")
                            tool_name = block.get("name", "unknown")

                            tool_call = ToolCallResponse(
                                call_id=call_id,
                                event_type=ToolCallEventType.CALL,
                                tool_name=tool_name,
                                tool_response=f"Running tool {tool_name}",
                                tool_call_details={},
                            )
                            active_tool_calls[call_id] = tool_call
                            partial_json_buffer[call_id] = ""

                            yield ChatAgentResponse(
                                response="",
                                tool_calls=[tool_call],
                                citations=[],
                            )

                    # --- content_block_delta ---
                    elif etype == "content_block_delta":
                        delta = event.get("delta", {})
                        dtype = delta.get("type")

                        if dtype == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                full_response += text
                                reasoning_mgr.append_content(text)
                                yield ChatAgentResponse(
                                    response=text,
                                    tool_calls=[],
                                    citations=[],
                                )

                        elif dtype == "input_json_delta":
                            chunk = delta.get("partial_json", "")
                            for cid in list(active_tool_calls.keys()):
                                if cid in partial_json_buffer:
                                    partial_json_buffer[cid] += chunk
                                    try:
                                        active_tool_calls[cid].tool_call_details = json.loads(
                                            partial_json_buffer[cid]
                                        )
                                    except json.JSONDecodeError:
                                        pass  # Still streaming

                    # --- content_block_stop ---
                    elif etype == "content_block_stop":
                        for cid, buf in partial_json_buffer.items():
                            if buf:
                                try:
                                    active_tool_calls[cid].tool_call_details = json.loads(buf)
                                except json.JSONDecodeError:
                                    pass

                    # --- message_delta ---
                    elif etype == "message_delta":
                        delta = event.get("delta", {})
                        if "reasoning_hash" in delta:
                            yield ChatAgentResponse(
                                response="",
                                tool_calls=[],
                                citations=[],
                                reasoning_hash=delta["reasoning_hash"],
                            )

                elif isinstance(message, ResultMessage):
                    # Extract citations from full response
                    citations = self._extract_citations(full_response)
                    if citations:
                        yield ChatAgentResponse(
                            response="",
                            tool_calls=[],
                            citations=citations,
                        )

                    # Check for reasoning hash in result
                    if hasattr(message, "reasoning_hash") and message.reasoning_hash:
                        yield ChatAgentResponse(
                            response="",
                            tool_calls=[],
                            citations=[],
                            reasoning_hash=message.reasoning_hash,
                        )

                else:
                    # Handle tool results from historical messages
                    self._handle_tool_results_from_message(message, active_tool_calls)

        except Exception as e:
            logger.error(f"Claude SDK stream error: {e}")
            yield ChatAgentResponse(
                response=f"\n\n[Service Alert: The stream was interrupted. Error: {str(e)}]",
                tool_calls=[],
                citations=[],
            )

        logger.info("claude-agent-sdk stream completed")

    # ------------------------------------------------------------------
    # Internal event processors
    # ------------------------------------------------------------------

    def _process_stream_event(
        self,
        message: StreamEvent,
        agent_response: ChatAgentResponse,
        active_tool_calls: Dict[str, ToolCallResponse],
        partial_json_buffer: Dict[str, str],
    ) -> None:
        """Map a StreamEvent into the ChatAgentResponse accumulator."""
        event = message.event
        etype = event.get("type")

        if etype == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                call_id = block.get("id", "")
                tool_name = block.get("name", "unknown")
                tool_call = ToolCallResponse(
                    call_id=call_id,
                    event_type=ToolCallEventType.CALL,
                    tool_name=tool_name,
                    tool_response=f"Running tool {tool_name}",
                    tool_call_details={},
                )
                active_tool_calls[call_id] = tool_call
                agent_response.tool_calls.append(tool_call)
                partial_json_buffer[call_id] = ""

        elif etype == "content_block_delta":
            delta = event.get("delta", {})
            dtype = delta.get("type")

            if dtype == "text_delta":
                text = delta.get("text", "")
                agent_response.response += text

            elif dtype == "input_json_delta":
                chunk = delta.get("partial_json", "")
                for cid in active_tool_calls:
                    if cid in partial_json_buffer:
                        partial_json_buffer[cid] += chunk
                        try:
                            active_tool_calls[cid].tool_call_details = json.loads(
                                partial_json_buffer[cid]
                            )
                        except json.JSONDecodeError:
                            pass

        elif etype == "content_block_stop":
            for cid, buf in partial_json_buffer.items():
                if buf:
                    try:
                        active_tool_calls[cid].tool_call_details = json.loads(buf)
                    except json.JSONDecodeError:
                        pass

        elif etype == "message_delta":
            delta = event.get("delta", {})
            if "reasoning_hash" in delta:
                agent_response.reasoning_hash = delta["reasoning_hash"]

    def _process_result_message(
        self, message: ResultMessage, agent_response: ChatAgentResponse
    ) -> None:
        """Handle ResultMessage — extract citations and reasoning hash."""
        citations = self._extract_citations(agent_response.response)
        if citations:
            agent_response.citations = sorted(
                set(agent_response.citations + citations)
            )

        if hasattr(message, "reasoning_hash") and message.reasoning_hash:
            agent_response.reasoning_hash = message.reasoning_hash

    def _process_other_message(self, message, agent_response: ChatAgentResponse) -> None:
        """Handle historical turn messages — look for tool_result blocks."""
        self._handle_tool_results_from_message(
            message,
            {tc.call_id: tc for tc in agent_response.tool_calls},
        )

    @staticmethod
    def _handle_tool_results_from_message(
        message, active_tool_calls: Dict[str, ToolCallResponse]
    ) -> None:
        """Extract tool results from SDK historical messages."""
        msg_data = getattr(message, "__dict__", message)
        if isinstance(msg_data, dict) and "content" in msg_data:
            for block in msg_data["content"]:
                b_type = (
                    getattr(block, "type", None)
                    or (block.get("type") if isinstance(block, dict) else None)
                )
                if b_type == "tool_result":
                    tool_use_id = (
                        getattr(block, "tool_use_id", None)
                        or (block.get("tool_use_id") if isinstance(block, dict) else None)
                    )
                    if tool_use_id and tool_use_id in active_tool_calls:
                        tc = active_tool_calls[tool_use_id]
                        tc.event_type = ToolCallEventType.RESULT
                        tc.tool_response = str(
                            getattr(block, "content", None)
                            or (block.get("content", "") if isinstance(block, dict) else "")
                        )
