from typing import AsyncGenerator, TYPE_CHECKING
if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class SalesPerformanceAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Sales Performance Analyst",
            goal="Analyze and report on sales performance metrics, quota attainment, and revenue generation.",
            backstory=(
                "You are an expert Sales Performance Analyst working under Gabrial, the Chief Revenue Officer for Railvision. "
                "You analyze complex sales data, track KPIs like win rates, sales cycles, and quota attainment, "
                "and provide actionable insights to improve sales outcomes and team performance."
            ),
            tasks=[
                TaskConfig(
                    description="Analyze sales performance data and provide insights.",
                    expected_output="Detailed sales performance report and recommendations.",
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
