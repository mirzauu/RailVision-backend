from typing import AsyncGenerator, TYPE_CHECKING
if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class PipelineAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Sales Pipeline Manager",
            goal="Manage, forecast, and optimize the sales pipeline and conversion rates.",
            backstory=(
                "You are a specialized Sales Pipeline Manager working under Gabrial, the Chief Revenue Officer for Railvision. "
                "You monitor the health of the sales pipeline, forecast future revenue, identify bottlenecks in the sales process, "
                "and ensure smooth progression of deals from lead to closed-won."
            ),
            tasks=[
                TaskConfig(
                    description="Review pipeline health and forecast revenue.",
                    expected_output="Pipeline analysis, forecasts, and optimization strategies.",
                )
            ],
        )
        tools = self.tools_provider.get_tools(["web_search_tool", "knowledge_base", "search_attachments"]) if self.tools_provider else []
        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk
