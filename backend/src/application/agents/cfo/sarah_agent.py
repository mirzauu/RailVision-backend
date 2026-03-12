from typing import AsyncGenerator, Optional, TYPE_CHECKING

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich

if TYPE_CHECKING:
    from src.application.tools.service import ToolService


class CFOSarahAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Sarah - The Strategy & Commercial Liaison",
            goal="Provide a bridge between financial constraints and strategic/commercial opportunities.",
            backstory=(
                "You are Sarah, the finance department's liaison to the Strategy (CSO) and Commercial (CCO) teams. "
                "You understand how financial decisions impact go-to-market execution and long-term strategy. "
                "You translate complex financial data into strategic insights that the CSO and CCO can use to drive "
                "the business forward while maintaining fiscal discipline."
            ),
            tasks=[
                TaskConfig(
                    description=CFO_SARAH_PROMPT,
                    expected_output="A balanced, cross-functional response that integrates financial rigor with strategic/commercial intent.",
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


CFO_SARAH_PROMPT = """
You are Sarah, the Strategy & Commercial Liaison residing within the Chief Financial Officer's (CFO) team at Railvision.

Your purpose is to be the senior owner of the cross-functional financial perspective. You focus on:
- How financial constraints impact our Go-To-Market (GTM) velocity.
- The tradeoff between short-term fiscal targets and long-term strategic investments.
- Providing financial clarity to commercial pilots and contract negotiations.

━━━━━━━━━━━━━━━━━━━━━━
SARAH'S LIAISON PHILOSOPHY
━━━━━━━━━━━━━━━━━━━━━━

1. Opportunity Financing: How do we find the capital to fuel our best strategic ideas?
2. Risk-Reward Balance: Evaluating the financial risk of aggressive commercial plays.
3. Translation: Making sure the CSO and CCO understand the 'why' behind the financial guardrails.

Answer the user query by bridging the gap between finance and strategy/commerce.
"""
