from typing import AsyncGenerator, Optional, TYPE_CHECKING

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import (
    AgentConfig,
    ChatAgent,
    ChatAgentResponse,
    ChatContext,
    TaskConfig,
)
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich

if TYPE_CHECKING:
    from src.application.tools.service import ToolService


class CFOSpreadsheetAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: Optional["ToolService"] = None,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CFO Spreadsheet Specialist",
            goal=(
                "Generate clean, well-structured Excel spreadsheets from financial data "
                "requests, returning a download link for the user."
            ),
            backstory=(
                "You are the financial modeling and reporting specialist at RailVision. "
                "You transform structured financial requests (budgets, cash flows, P&L, "
                "valuation models) into professional Excel workbooks "
                "with multiple sheets, clear column headers, and sensible layouts. "
                "You never show raw data as text — you always produce a downloadable Excel file."
            ),
            tasks=[
                TaskConfig(
                    description=CFO_SPREADSHEET_PROMPT,
                    expected_output=(
                        "A fully formed Excel (.xlsx) file with the requested sheets and data, "
                        "plus a download link returned to the user."
                    ),
                )
            ],
        )

        tools = (
            self.tools_provider.get_tools(
                [
                    "think",
                    "knowledge_base",
                    "create_spreadsheet",
                    "get_spreadsheet_link",
                    "search_attachments",
                    "create_todo",
                    "update_todo_status",
                    "add_todo_note",
                    "get_todo",
                    "list_todos",
                    "get_todo_summary",
                ]
            )
            if self.tools_provider
            else []
        )

        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        enriched_query = (
            await context_enrich(ctx.query, user_id=self.tools_provider.user_id)
            if self.tools_provider
            else ctx.query
        )
        new_ctx = ctx.model_copy(update={"query": enriched_query})
        return await self._build_agent().run(new_ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        enriched_query = (
            await context_enrich(ctx.query, user_id=self.tools_provider.user_id)
            if self.tools_provider
            else ctx.query
        )
        new_ctx = ctx.model_copy(update={"additional_context": enriched_query})
        async for chunk in self._build_agent().run_stream(new_ctx):
            yield chunk


CFO_SPREADSHEET_PROMPT = """
You are the Chief Financial Officer (CFO) Specialist in Spreadsheet Generation.

Your SOLE PURPOSE is to produce Excel spreadsheets using the `create_spreadsheet` tool.

━━━━━━━━━━━━━━━━━━━━━━
🚨 MANDATORY: TOOL-FIRST POLICY 🚨
━━━━━━━━━━━━━━━━━━━━━━

- **NEVER** paste data as a markdown table.
- **ALWAYS** follow this exact sequence:
    1. `think`: Plan the sheets and columns.
    2. `knowledge_base` or `search_attachments`: Retrieve factual financial data.
    3. `create_spreadsheet`: Call with the structured data.

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━

After the tool runs, provide:
1. Short confirmation.
2. Download link.
3. Brief one-line summary.

Produce the spreadsheet using your TOOLS now.
"""
