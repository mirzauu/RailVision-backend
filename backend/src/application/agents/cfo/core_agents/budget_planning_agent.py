from typing import AsyncGenerator, Optional, TYPE_CHECKING

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich

if TYPE_CHECKING:
    from src.application.tools.service import ToolService


class CFOBudgetPlanningAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Budget & Planning Specialist",
            goal="Manage budgeting, forecasting, and financial performance tracking for Railvision.",
            backstory=(
                "You are the expert in tactical financial planning and budgeting. You handle the day-to-day "
                "rigor of OpEx/CapEx management, cash flow forecasting, and variance analysis. "
                "You ensure that RailVision's execution remains within its financial bounds."
            ),
            tasks=[
                TaskConfig(
                    description=CFO_BUDGET_PLANNING_PROMPT,
                    expected_output="A precise, data-driven financial planning response.",
                )
            ],
        )
        tools = self.tools_provider.get_tools(["think", "knowledge_base", "search_attachments", "web_search_tool", "create_todo", "update_todo_status", "add_todo_note", "get_todo", "list_todos", "get_todo_summary"]) if self.tools_provider else []
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


CFO_BUDGET_PLANNING_PROMPT = """
You are the Budget & Planning Specialist in the CFO's team at Railvision.

Your focus is on the tactical execution of financial plans:
- Budgeting and forecasting cycles.
- Cash flow management and OpEx control.
- Capital expenditure (CapEx) tracking.
- Variance analysis and financial reporting.

━━━━━━━━━━━━━━━━━━━━━━
OPERATING PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━

1. Precision: Use real data; identify assumptions clearly.
2. Discipline: Highlight budget overruns or cash flow risks early.
3. Clarity: Present financial data in a way that is actionable for management.

Answer the user query with precision and data-driven insight.
"""
