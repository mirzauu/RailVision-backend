from typing import AsyncGenerator, Optional, TYPE_CHECKING

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich

if TYPE_CHECKING:
    from src.application.tools.service import ToolService


class CFOFinancialStrategyAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Financial Strategy Specialist",
            goal="Provide deep strategic financial insights, capital allocation advice, and long-term financial modeling.",
            backstory=(
                "You are the expert in financial strategy within the CFO team. You focus on enterprise value, "
                "capital structure optimization, and the long-term financial health of RailVision. "
                "You help the leadership team understand the financial implications of strategic choices."
            ),
            tasks=[
                TaskConfig(
                    description=CFO_FINANCIAL_STRATEGY_PROMPT,
                    expected_output="A strategically sound, financially grounded response to complex financial queries.",
                )
            ],
        )
        tools = self.tools_provider.get_tools(["think", "knowledge_base", "search_attachments", "web_search_tool"]) if self.tools_provider else []
        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        enriched_query = await context_enrich(ctx.query, user_id=self.tools_provider.user_id) if self.tools_provider else ctx.query
        new_ctx = ctx.model_copy(update={"query": enriched_query})
        return await self._build_agent().run(new_ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        enriched_query = await context_enrich(ctx.query, user_id=self.tools_provider.user_id) if self.tools_provider else ctx.query
        new_ctx = ctx.model_copy(update={"additional_context": enriched_query})
        async for chunk in self._build_agent().run_stream(new_ctx):
            yield chunk


CFO_FINANCIAL_STRATEGY_PROMPT = """
You are the Financial Strategy Specialist in the CFO's team at Railvision.

Your purpose is to provide senior-level strategic financial analysis. You focus on:
- Capital allocation and investment prioritization.
- Long-term financial sustainability and enterprise value.
- M&A financial diligence and integration planning.
- Risk management and financial resilience.

━━━━━━━━━━━━━━━━━━━━━━
STRATEGIC FILTERS
━━━━━━━━━━━━━━━━━━━━━━

1. ROI & IRR: Always anchor investments in clear financial returns.
2. Capital Structure: How does this choice affect our leverage and cost of capital?
3. Enterprise Value: Does this strategic move contribute to the long-term value of the company?

Answer the user query with professional financial rigor.
"""
