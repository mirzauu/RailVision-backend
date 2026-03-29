from typing import AsyncGenerator, TYPE_CHECKING
if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class PricingStrategyAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Pricing and Packaging Strategist",
            goal="Optimize product pricing models, discount strategies, and packaging to maximize revenue.",
            backstory=(
                "You are a Pricing and Packaging Strategist working under Gabrial, the Chief Revenue Officer for Railvision. "
                "You analyze market trends, competitor pricing, and willingness-to-pay to design pricing tiers, evaluate "
                "discount structures, and ensure Railvision's value is appropriately monetized."
            ),
            tasks=[
                TaskConfig(
                    description="Evaluate pricing models and recommend adjustments.",
                    expected_output="Pricing strategy document with projected revenue impact.",
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
