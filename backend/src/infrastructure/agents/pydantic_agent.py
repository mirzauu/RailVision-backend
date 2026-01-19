import logging
import re
from typing import AsyncGenerator, List, Dict

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
)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider

from src.infrastructure.llm.provider_service import ProviderService
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
            model = OpenAIModel(model_name=model_id, provider=OpenAIProvider(api_key=api_key, base_url=base_url))
        elif provider == "anthropic":
            model = AnthropicModel(model_name=model_id, provider=AnthropicProvider(api_key=api_key, base_url=base_url))
        else:
            model = OpenAIModel(model_name=model_id, provider=OpenAIProvider(api_key=api_key, base_url=base_url))

        supports_tools = provider_config.capabilities.get("supports_tool_parallelism", True)
        if not supports_tools:
            final_tools = []

        model_settings = {"max_tokens": 8000}
        if final_tools and len(final_tools) > 0:
            model_settings["parallel_tool_calls"] = True

        self.agent = PydanticAgent(
            model=model,
            tools=final_tools,
            system_prompt=f"Role: {config.role}\nGoal: {config.goal}\nBackstory: {config.backstory}. Respond to the user query",
            retries=3,
            defer_model_check=True,
            end_strategy="exhaustive",
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
        return ChatAgentResponse(response=response_text, tool_calls=[], citations=[])

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        task = self._create_task_description(self.tasks[0], ctx)
        message_history, extra_user_content = self._build_history(ctx.history)
        if extra_user_content:
            task = f"Additional User Context:\n{extra_user_content}\n\n{task}"
        async with self.agent.iter(
            user_prompt=task,
            message_history=message_history,
        ) as run:
            async for node in run:
                if PydanticAgent.is_model_request_node(node):
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                                yield ChatAgentResponse(response=event.part.content, tool_calls=[], citations=[])
                            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                yield ChatAgentResponse(response=event.delta.content_delta, tool_calls=[], citations=[])
                elif PydanticAgent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as handle_stream:
                        async for event in handle_stream:
                            if isinstance(event, FunctionToolCallEvent):
                                yield ChatAgentResponse(
                                    response="",
                                    tool_calls=[
                                        ToolCallResponse(
                                            call_id=event.part.tool_call_id or "",
                                            event_type=ToolCallEventType.CALL,
                                            tool_name=event.part.tool_name,
                                            tool_response=f"Running tool {event.part.tool_name}",
                                            tool_call_details={
                                                "summary": {"tool": event.part.tool_name, "args": event.part.args_as_dict()}
                                            },
                                        )
                                    ],
                                    citations=[],
                                )
                            if isinstance(event, FunctionToolResultEvent):
                                yield ChatAgentResponse(
                                    response="",
                                    tool_calls=[
                                        ToolCallResponse(
                                            call_id=event.result.tool_call_id or "",
                                            event_type=ToolCallEventType.RESULT,
                                            tool_name=event.result.tool_name or "unknown tool",
                                            tool_response=f"Completed tool {event.result.tool_name or 'unknown tool'}",
                                            tool_call_details={
                                                "summary": {"tool": event.result.tool_name or "unknown tool", "result": event.result.content}
                                            },
                                        )
                                    ],
                                    citations=[],
                                )
                elif PydanticAgent.is_end_node(node):
                    logger.info("result streamed successfully")
