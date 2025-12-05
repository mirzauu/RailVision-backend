import logging
import re
from typing import AsyncGenerator, List

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import Tool
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider

from src.infrastructure.llm.provider_service import ProviderService
from .base import (
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

        tools = tools or []
        for i, tool in enumerate(tools):
            tools[i].name = re.sub(r" ", "", tool.name)

        provider = llm_provider.chat_config.provider
        api_key = llm_provider._get_api_key(llm_provider.chat_config.auth_provider)

        if provider == "openai":
            model = OpenAIModel(
                model_name=llm_provider.chat_config.model,
                provider=OpenAIProvider(api_key=api_key),
            )
        elif provider == "anthropic":
            model = AnthropicModel(
                model_name=llm_provider.chat_config.model,
                provider=AnthropicProvider(api_key=api_key),
            )
        else:
            model = OpenAIModel(
                model_name=llm_provider.chat_config.model,
                provider=OpenAIProvider(api_key=api_key),
            )

        self.agent = PydanticAgent(
            model=model,
            tools=[Tool(name=t.name, description=t.description, function=t.function) for t in tools],
            system_prompt=f"Role: {config.role}\nGoal: {config.goal}\nBackstory: {config.backstory}. Respond to the user query",
            result_type=str,
            retries=3,
            defer_model_check=True,
            end_strategy="exhaustive",
            model_settings={"parallel_tool_calls": True, "max_tokens": 8000},
        )

    def _create_task_description(self, task_config: TaskConfig, ctx: ChatContext) -> str:
        return (
            f"\n                CONTEXT:\n                User Query: {ctx.query}\n                Project ID: {ctx.project_id}\n                \n                Additional Context:\n                {ctx.additional_context if ctx.additional_context != '' else 'no additional context'}\n\n                TASK:\n                {task_config.description}\n\n                Expected Output:\n                {task_config.expected_output}\n\n                INSTRUCTIONS:\n                1. Use the available tools to gather information\n                2. Process and synthesize the gathered information\n                3. Format your response in markdown, make sure it's well formatted\n                4. Include relevant code snippets and file references\n                5. Provide clear explanations\n                6. Verify your output before submitting\n\n                IMPORTANT:\n                - Use tools efficiently and avoid unnecessary API calls\n                - Only use the tools listed below\n\n                With above information answer the user query: {ctx.query}\n            "
        )

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        logger.info("running pydantic-ai agent")
        task = self._create_task_description(self.tasks[0], ctx)
        resp = await self.agent.run(user_prompt=task)
        return ChatAgentResponse(response=resp.data, tool_calls=[], citations=[])

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        task = self._create_task_description(self.tasks[0], ctx)
        async with self.agent.iter(
            user_prompt=task,
            message_history=[ModelResponse([TextPart(content=msg)]) for msg in ctx.history],
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
                                                "summary": {
                                                    "tool": event.part.tool_name,
                                                    "args": event.part.args_as_dict(),
                                                }
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
                                                "summary": {
                                                    "tool": event.result.tool_name or "unknown tool",
                                                    "result": event.result.content,
                                                }
                                            },
                                        )
                                    ],
                                    citations=[],
                                )
                elif PydanticAgent.is_end_node(node):
                    logger.info("result streamed successfully")

