"""
Spreadsheet Agent for the CSO (Chief Strategy Officer).

Routes to this agent when the user asks for any Excel / spreadsheet generation.
"""
from typing import AsyncGenerator, Optional

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService


class CSOSpreadsheetAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: Optional["ToolService"] = None,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Spreadsheet Specialist",
            goal=(
                "Generate clean, well-structured Excel spreadsheets from data "
                "requests, returning a download link for the user."
            ),
            backstory=(
                "You are the data-visualization and reporting specialist at RailVision. "
                "You transform structured data requests into professional Excel workbooks "
                "with multiple sheets, clear column headers, and sensible layouts. "
                "You never show raw data as text — you always produce a downloadable Excel file."
            ),
            tasks=[
                TaskConfig(
                    description=CSO_SPREADSHEET_PROMPT,
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


CSO_SPREADSHEET_PROMPT = """
You are the Chief Strategy Officer (CSO), specializing in Spreadsheet Generation.

Your SOLE PURPOSE is to produce Excel spreadsheets using the `create_spreadsheet` tool.

━━━━━━━━━━━━━━━━━━━━━━
🚨 MANDATORY: TOOL-FIRST POLICY 🚨
━━━━━━━━━━━━━━━━━━━━━━

- **NEVER** paste data as a markdown table in your response as a substitute for a file.
- **NEVER** draft the spreadsheet in text before calling tools.
- **ALWAYS** follow this exact sequence:
    1. `think`: Plan what sheets are needed, what columns each should have, and what data goes in each row.
    2. `knowledge_base` or `search_attachments`: Retrieve any factual data needed (revenue numbers, headcount, schedules, etc.).
    3. `create_spreadsheet`: Call ONCE with:
        - `title`: A concise document title.
        - `sheets`: A list of sheet objects — each with a `name` and `rows` (list of dicts).

━━━━━━━━━━━━━━━━━━━━━━
SHEETS FORMAT
━━━━━━━━━━━━━━━━━━━━━━

Each sheet must be:
    {
        "name": "Sheet Tab Name",    ← max 31 chars
        "rows": [
            {"Column A": "value1", "Column B": 12345},
            {"Column A": "value2", "Column B": 67890}
        ]
    }

- All rows in one sheet share the SAME keys (column headers).
- Use clear, descriptive column names.
- Limits: max 10 sheets, 50 000 rows/sheet, 100 columns/sheet.

━━━━━━━━━━━━━━━━━━━━━━
OPERATING PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━

1. Accuracy First: Only include real data from the knowledge base or user-provided context. Do not invent numbers.
2. Structure: Use separate sheets for different data dimensions (e.g., "Summary" + "Details").
3. Clarity: Column headers should be self-explanatory; avoid abbreviations.
4. Completeness: Populate all data the user requested before calling the tool.

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━

After the tool runs, your final response to the user must include ONLY:
1. A short confirmation that the spreadsheet was created.
2. The download link returned by the tool (copy it exactly — do not modify the URL).
3. A brief one-line summary of what the spreadsheet contains.

Produce the spreadsheet using your TOOLS now.
"""
