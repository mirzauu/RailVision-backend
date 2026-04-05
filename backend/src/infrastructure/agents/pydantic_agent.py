import logging
import re
from typing import AsyncGenerator, List, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import Tool
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    UserPromptPart,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
import asyncio
import httpx

from src.infrastructure.llm.provider_service import ProviderService
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
from src.infrastructure.agents.tools.tool_progress import (
    get_queue as _get_progress_queue,
    DONE_SENTINEL as _QUEUE_DONE_SENTINEL,
    get_all_active_tool_names as _get_active_tools,
)
# Also import doc-gen sentinel for backward compat (ppt/word use their own registry)
from src.infrastructure.agents.tools.claude_document_generator import (
    _QUEUE_DONE as _DOC_GEN_DONE_SENTINEL,
)

logger = logging.getLogger(__name__)

# Tool names for which we stream progress (all tools using tool_progress registry)
# This set is used only to detect doc-gen tools that use the OLD queue system
_LEGACY_DOC_GEN_TOOL_NAMES = {"create_ppt", "create_word_doc"}


def _get_legacy_doc_queue(tool_name: str):
    """Look up the active progress queue for legacy doc-gen tools (ppt/word)."""
    if tool_name == "create_ppt":
        from src.infrastructure.agents.tools.ppt_tool import get_active_progress_queue
        return get_active_progress_queue(tool_name), _DOC_GEN_DONE_SENTINEL
    elif tool_name == "create_word_doc":
        from src.infrastructure.agents.tools.word_tool import get_active_progress_queue
        return get_active_progress_queue(tool_name), _DOC_GEN_DONE_SENTINEL
    return None, None


def _get_any_progress_queue(tool_name: str):
    """Return (queue, done_sentinel) for any tool — central registry first, then legacy."""
    # 1. Central registry (think, knowledge_base, web_search, attachments, etc.)
    q = _get_progress_queue(tool_name)
    if q is not None:
        return q, _QUEUE_DONE_SENTINEL
    # 2. Legacy doc-gen tools (ppt, word) use their own module-level queues
    q, sentinel = _get_legacy_doc_queue(tool_name)
    return q, sentinel



class PydanticChatAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[Tool] | None = None,
    ):
        self.tasks = config.tasks
        self.max_iter = config.max_iter

        final_tools = []
        tools = tools or []
        for tool in tools:
            clean_name = re.sub(r" ", "", tool.name)
            
            if isinstance(tool, StructuredTool):
                func = tool.coroutine if tool.coroutine else tool.func
                final_tools.append(Tool(func, name=clean_name, description=tool.description))
            else:
                if hasattr(tool, "name"):
                    try:
                        tool.name = clean_name
                    except AttributeError:
                        pass
                final_tools.append(tool)

        provider_config = llm_provider.chat_config
        provider = provider_config.provider
        auth_provider = provider_config.auth_provider
        api_key = llm_provider._get_api_key(auth_provider)
        base_url = provider_config.base_url
        if not api_key and provider == "openai":
            try:
                from src.config.settings import settings
                if getattr(settings, "openai_api_key", None):
                    api_key = settings.openai_api_key
            except Exception:
                pass
            if not api_key:
                import os
                api_key = os.environ.get("OPENAI_API_KEY")

        model_id = provider_config.model.split("/")[-1]

        if provider == "openai":
            model = OpenAIModel(model_name=model_id, provider=OpenAIProvider(api_key=api_key, base_url=base_url, http_client=httpx.AsyncClient(timeout=600.0)))
        elif provider == "anthropic":
            # Use a more stable initialization for the provider
            model = AnthropicModel(
                model_name=model_id, 
                provider=AnthropicProvider(
                    api_key=api_key, 
                    http_client=httpx.AsyncClient(timeout=600.0)
                )
            )
        else:
            model = OpenAIModel(model_name=model_id, provider=OpenAIProvider(api_key=api_key, base_url=base_url))

        supports_tools = provider_config.capabilities.get("supports_tool_parallelism", True)
        if not supports_tools:
            final_tools = []

        model_settings = {"max_tokens": provider_config.default_params.get("max_tokens", 8000)}
        if final_tools and len(final_tools) > 0:
            # Only enable parallel tool calls for non-anthropic providers if stability is an issue
            if provider != "anthropic":
                model_settings["parallel_tool_calls"] = True

        self.agent = PydanticAgent(
            model=model,
            tools=final_tools,
            system_prompt=f"Role: {config.role}\nGoal: {config.goal}\nBackstory: {config.backstory}. Respond to the user query",
            retries=3,
            defer_model_check=True,
            model_settings=model_settings,
        )

    def _create_task_description(self, task_config: TaskConfig, ctx: ChatContext) -> str:
        return (
            f"\n                CONTEXT:\n                User Query: {ctx.query}\n                \n                Additional Context:\n                {ctx.additional_context if ctx.additional_context != '' else 'no additional context'}\n\n                TASK:\n                {task_config.description}\n\n                Expected Output:\n                {task_config.expected_output}\n\n                INSTRUCTIONS:\n                1. Use the available tools to gather information\n                2. Process and synthesize the gathered information\n                3. Format your response in markdown, make sure it's well formatted\n                4. Include relevant code snippets and file references\n                5. Provide clear explanations\n                6. Verify your output before submitting\n\n                IMPORTANT:\n                - Use tools efficiently and avoid unnecessary API calls\n                - Only use the tools listed below\n\n                With above information answer the user query: {ctx.query}\n            "
        )

    def _build_history(self, history: List[Dict[str, str]]) -> tuple[List[ModelRequest | ModelResponse], str | None]:
        if not history:
            return [], None

        final_msgs = []
        last_role = None
        current_parts = []

        for m in history:
            role = m.get("role")
            content = m.get("content", "")
            if role == last_role:
                current_parts.append(content)
            else:
                if last_role:
                    merged_content = "\n\n".join(current_parts)
                    if last_role == "user":
                        final_msgs.append(ModelRequest([UserPromptPart(content=merged_content)]))
                    else:
                        final_msgs.append(ModelResponse([TextPart(content=merged_content)]))
                current_parts = [content]
                last_role = role

        # Process the last group
        last_user_content = None
        if last_role == "user":
            last_user_content = "\n\n".join(current_parts)
        elif last_role == "assistant":
            merged_content = "\n\n".join(current_parts)
            final_msgs.append(ModelResponse([TextPart(content=merged_content)]))

        return final_msgs, last_user_content
    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        logger.info("running pydantic-ai agent")
        task = self._create_task_description(self.tasks[0], ctx)
        message_history, extra_user_content = self._build_history(ctx.history)
        if extra_user_content:
            task = f"Additional User Context:\n{extra_user_content}\n\n{task}"
        resp = await self.agent.run(user_prompt=task, message_history=message_history)
        
        # Extract response text
        response_text = None
        if isinstance(resp, str):
            response_text = resp
        else:
            for attr in ("text", "output_text", "response_text"):
                value = getattr(resp, attr, None)
                if callable(value):
                    try:
                        response_text = value()
                        break
                    except Exception:
                        pass
                elif isinstance(value, str):
                    response_text = value
                    break
            if response_text is None:
                value_attr = getattr(resp, "value", None)
                if isinstance(value_attr, str):
                    response_text = value_attr
        if response_text is None:
            response_text = str(resp)

        # Extract tool calls from message history
        tool_calls = []
        try:
            from pydantic_ai.messages import ModelResponse, ToolCallPart, ModelRequest, ToolReturnPart
            
            # Look at new messages in this run
            for msg in resp.new_messages():
                if isinstance(msg, ModelResponse):
                    for part in msg.parts:
                        if isinstance(part, ToolCallPart):
                            # Safely extract arguments
                            args = {}
                            try:
                                args = part.args_as_dict()
                            except Exception:
                                # Fallback to raw args attribute if it exists and is already a dict or can be used
                                args = getattr(part, 'args', {})
                                if not isinstance(args, dict):
                                    args = {"raw_args": str(args)}

                            tool_calls.append(
                                ToolCallResponse(
                                    call_id=part.tool_call_id or "",
                                    event_type=ToolCallEventType.CALL,
                                    tool_name=part.tool_name,
                                    tool_response=f"Running tool {part.tool_name}",
                                    tool_call_details={
                                        "summary": {"tool": part.tool_name, "args": args}
                                    },
                                )
                            )
                elif isinstance(msg, ModelRequest):
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart):
                            result_content = part.content if isinstance(part.content, str) else str(part.content)
                            tool_calls.append(
                                ToolCallResponse(
                                    call_id=part.tool_call_id or "",
                                    event_type=ToolCallEventType.RESULT,
                                    tool_name=part.tool_name or "unknown tool",
                                    tool_response=result_content,
                                    tool_call_details={
                                        "summary": {"tool": part.tool_name or "unknown tool", "result": part.content}
                                    },
                                )
                            )
        except Exception as e:
            logger.warning(f"Failed to extract tool calls in run(): {e}")

        return ChatAgentResponse(response=response_text, tool_calls=tool_calls, citations=[])

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        task = self._create_task_description(self.tasks[0], ctx)
        message_history, extra_user_content = self._build_history(ctx.history)
        if extra_user_content:
            task = f"Additional User Context:\n{extra_user_content}\n\n{task}"
        try:
            async with self.agent.iter(
                user_prompt=task,
                message_history=message_history,
            ) as run:
                async for node in run:
                    logger.info(f"Stepping into stream node: {type(node).__name__}")
                    if PydanticAgent.is_model_request_node(node):
                        reasoning_mgr = get_reasoning_manager()
                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:
                                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                                    reasoning_mgr.append_content(event.part.content)
                                    yield ChatAgentResponse(response=event.part.content, tool_calls=[], citations=[])
                                if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                    reasoning_mgr.append_content(event.delta.content_delta)
                                    yield ChatAgentResponse(response=event.delta.content_delta, tool_calls=[], citations=[])
                                if isinstance(event, PartDeltaEvent) and isinstance(event.delta, ToolCallPartDelta):
                                    if event.delta.tool_name_delta:
                                        yield ChatAgentResponse(response=f"\n[Thinking: {event.delta.tool_name_delta}]", tool_calls=[], citations=[])
                    elif PydanticAgent.is_call_tools_node(node):
                        # Universal real-time progress streaming for ALL tools.
                        # Any tool that calls begin_tool/push_progress/end_tool from
                        # tool_progress.py is automatically streamed here.
                        #
                        # Strategy:
                        #   1. Run the pydantic-ai tool node stream in a background task
                        #      (it pushes FunctionToolCallEvent / FunctionToolResultEvent
                        #      into tool_event_queue as they arrive)
                        #   2. Main loop polls both the tool_event_queue (100ms timeout)
                        #      AND the per-tool progress queue, yielding PROGRESS chunks

                        tool_event_queue: asyncio.Queue = asyncio.Queue()
                        # tool_name → call_id for currently running tools
                        active_tools: dict = {}

                        async def _collect_tool_events():
                            """Background task: collect pydantic-ai node events into a queue."""
                            try:
                                async with node.stream(run.ctx) as handle_stream:
                                    async for event in handle_stream:
                                        await tool_event_queue.put(event)
                            except Exception as exc:
                                logger.error(f"Tool event collector error: {exc}")
                            finally:
                                await tool_event_queue.put(None)  # stream-done sentinel

                        collector_task = asyncio.create_task(_collect_tool_events())

                        def _drain_progress(tool_name: str, call_id: str):
                            """Inner generator — yield PROGRESS chunks from a tool's queue."""
                            progress_q, done_sentinel = _get_any_progress_queue(tool_name)
                            if progress_q is None:
                                return
                            while not progress_q.empty():
                                try:
                                    chunk = progress_q.get_nowait()
                                    if chunk is done_sentinel:
                                        active_tools.pop(tool_name, None)
                                        break
                                    if isinstance(chunk, str) and chunk:
                                        yield ChatAgentResponse(
                                            response="",
                                            tool_calls=[
                                                ToolCallResponse(
                                                    call_id=call_id,
                                                    event_type=ToolCallEventType.PROGRESS,
                                                    tool_name=tool_name,
                                                    tool_response=chunk,
                                                    tool_call_details={},
                                                )
                                            ],
                                            citations=[],
                                        )
                                except Exception:
                                    break

                        try:
                            while True:
                                # 1. Drain progress from all currently active tools
                                for t_name, t_call_id in list(active_tools.items()):
                                    for chunk_resp in _drain_progress(t_name, t_call_id):
                                        yield chunk_resp

                                # 2. Poll tool event queue (100 ms timeout keeps the loop alive)
                                try:
                                    event = await asyncio.wait_for(
                                        tool_event_queue.get(), timeout=0.1
                                    )
                                except asyncio.TimeoutError:
                                    continue  # no new pydantic-ai event yet, loop back

                                if event is None:
                                    # pydantic-ai stream finished — final drain for all tools
                                    for t_name, t_call_id in list(active_tools.items()):
                                        for chunk_resp in _drain_progress(t_name, t_call_id):
                                            yield chunk_resp
                                    break

                                # 3. Process pydantic-ai tool lifecycle events
                                if isinstance(event, FunctionToolCallEvent):
                                    args = {}
                                    try:
                                        args = event.part.args_as_dict()
                                    except Exception:
                                        args = getattr(event.part, "args", {})
                                        if not isinstance(args, dict):
                                            args = {"raw_args": str(args)}

                                    t_name = event.part.tool_name
                                    t_call_id = event.part.tool_call_id or ""
                                    # Track this tool for progress polling
                                    active_tools[t_name] = t_call_id

                                    yield ChatAgentResponse(
                                        response="",
                                        tool_calls=[
                                            ToolCallResponse(
                                                call_id=t_call_id,
                                                event_type=ToolCallEventType.CALL,
                                                tool_name=t_name,
                                                tool_response=f"Running tool {t_name}",
                                                tool_call_details={
                                                    "summary": {"tool": t_name, "args": args}
                                                },
                                            )
                                        ],
                                        citations=[],
                                    )

                                elif isinstance(event, FunctionToolResultEvent):
                                    t_name = event.result.tool_name or "unknown tool"
                                    t_call_id = event.result.tool_call_id or ""
                                    call_id_for_drain = active_tools.get(t_name, t_call_id)

                                    # Final drain before emitting RESULT
                                    for chunk_resp in _drain_progress(t_name, call_id_for_drain):
                                        yield chunk_resp
                                    active_tools.pop(t_name, None)

                                    result_content = (
                                        event.result.content
                                        if isinstance(event.result.content, str)
                                        else str(event.result.content)
                                    )
                                    yield ChatAgentResponse(
                                        response="",
                                        tool_calls=[
                                            ToolCallResponse(
                                                call_id=t_call_id,
                                                event_type=ToolCallEventType.RESULT,
                                                tool_name=t_name,
                                                tool_response=result_content,
                                                tool_call_details={
                                                    "summary": {
                                                        "tool": t_name,
                                                        "result": event.result.content,
                                                    }
                                                },
                                            )
                                        ],
                                        citations=[],
                                    )
                        finally:
                            if not collector_task.done():
                                collector_task.cancel()
                                try:
                                    await collector_task
                                except asyncio.CancelledError:
                                    pass

                    elif PydanticAgent.is_end_node(node):
                        logger.info("result streamed successfully")
        except Exception as e:
            logger.error(f"Stream encountered a critical error: {e}")
            yield ChatAgentResponse(
                response=f"\n\n[Service Alert: The stream was interrupted. Error: {str(e)}]",
                tool_calls=[],
                citations=[]
            )
