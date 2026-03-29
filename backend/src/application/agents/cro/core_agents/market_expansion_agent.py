from typing import AsyncGenerator, TYPE_CHECKING
if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class MarketExpansionAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Market Expansion Strategist",
            goal="Identify and evaluate new market opportunities, territories, and revenue streams.",
            backstory=(
                "You are a Market Expansion Strategist working under Gabrial, the Chief Revenue Officer for Railvision. "
                "You research new geographies, emerging industries, and potential partnerships. You provide data-driven "
                "recommendations on where Railvision should allocate resources for maximum revenue growth."
            ),
            tasks=[
                TaskConfig(
                    description="Analyze new market opportunities and propose entry strategies.",
                    expected_output="Market expansion plan with revenue projections.",
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
