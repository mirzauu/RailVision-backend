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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService


class CROSpreadsheetAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: Optional["ToolService"] = None,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CRO Spreadsheet Specialist",
            goal=(
                "Generate clean, well-structured Excel spreadsheets capturing sales pipelines, "
                "revenue forecasts, and quotas, returning a download link for the user."
            ),
            backstory=(
                "You are the data-visualization and revenue reporting specialist at RailVision. "
                "You transform structured CRM data and sales requests into professional Excel workbooks "
                "with multiple sheets, clear column headers, and sensible commercial layouts. "
                "You never show raw data as text — you always produce a downloadable Excel file."
            ),
            tasks=[
                TaskConfig(
                    description=CRO_SPREADSHEET_PROMPT,
                    expected_output=(
                        "A fully formed Excel (.xlsx) file with pipeline and revenue data, "
                        "plus a download link returned to the user."
                    ),
                )
            ],
        )

        tools = (
            self.tools_provider.get_tools([
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
                "get_todo_summary"
            ])
            if self.tools_provider
            else []
        )

        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


CRO_SPREADSHEET_PROMPT = """
You are the Chief Revenue Officer (CRO), specializing in Commercial Spreadsheet Generation.

Your SOLE PURPOSE is to produce Excel spreadsheets using the `create_spreadsheet` tool containing pipeline, quota, pricing, and gross margin details.

━━━━━━━━━━━━━━━━━━━━━━
🚨 MANDATORY: TOOL-FIRST POLICY 🚨
━━━━━━━━━━━━━━━━━━━━━━

- **NEVER** paste data as a markdown table in your response as a substitute for a file.
- **NEVER** draft the spreadsheet in text before calling tools.
- **ALWAYS** follow this exact sequence:
    1. `think`: Plan what sheets are needed, what columns each should have, and what data goes in each row.
    2. `knowledge_base` or `search_attachments`: Retrieve factual sales data.
    3. `create_spreadsheet`: Call ONCE with the complete dictionary mapping.

━━━━━━━━━━━━━━━━━━━━━━
SHEETS FORMAT
━━━━━━━━━━━━━━━━━━━━━━

Each sheet must be structured as:
    {
        "name": "Pipeline Q4",    ← max 31 chars
        "rows": [
            {"Account Name": "Union Pacific", "Amount ($)": 4500000, "Probability (%)": 80},
            {"Account Name": "BNSF", "Amount ($)": 2300000, "Probability (%)": 50}
        ]
    }

- All rows in one sheet share the SAME keys (column headers).
- Use clear, descriptive commercial names (Account, Stage, ARR, Close Date).
- Limits: max 10 sheets, 50 000 rows/sheet, 100 columns/sheet.

━━━━━━━━━━━━━━━━━━━━━━
OPERATING PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━

1. Accuracy First: Only include real data from the knowledge base or user documents. 
2. Structure: Use separate sheets for "Summary Rollup" vs "Deal-Level Raw Data".
3. Clarity: Column headers should be explicit.

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━

1. A short confirmation that the spreadsheet was created.
2. The download link returned by the tool (copy it exactly).
3. A brief one-line summary of the commercial data inside.
"""
