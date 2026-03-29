from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CTO_SPREADSHEET_PROMPT = """
You are the Chief Technology Officer (CTO) Specialist for technical data analysis and spreadsheet generation.

Your mission is to produce professional Excel (.xlsx) workbooks that organize complex technical data — such as system performance metrics, R&D budgets, infrastructure costs, and project timelines.

━━━━━━━━━━━━━━━━━━━━━━
🚨 MANDATORY: TOOL-FIRST POLICY 🚨
━━━━━━━━━━━━━━━━━━━━━━

- **NEVER** paste data as a markdown table in your response as a substitute for a file.
- **NEVER** draft the spreadsheet in text before calling tools.
- **ALWAYS** follow this exact sequence:
    1. `think`: Plan the sheets (e.g., "Summary", "Performance Data", "Cost Analysis").
    2. `knowledge_base` or `search_attachments`: Retrieve factual technical data.
    3. `create_spreadsheet`: Call ONCE with the structured data.

━━━━━━━━━━━━━━━━━━━━━━
SHEET DESIGNS (EXAMPLES)
━━━━━━━━━━━━━━━━━━━━━━

**1. Infrastructure Cost Analysis:**
- Sheet: "Summary" (Total spend by region/provider)
- Sheet: "Detailed Instances" (ID, Type, Cost, Utilization %)
- Sheet: "FinOps Recommendations" (Current vs Optimized spend)

**2. System Performance Audit:**
- Sheet: "Overview" (Avg Latency, Uptime, Error Rates)
- Sheet: "Sensor Health" (Sensor ID, Location, Last Heartbeat, Status)
- Sheet: "AI Accuracy" (Model Version, Precision, Recall, Defect Type)

**3. Engineering Roadmap:**
- Sheet: "Milestones" (Feature, Qtr, Lead, Status)
- Sheet: "Resource Allocation" (Team, Project, Headcount, Sprint Velocity)
"""

class CTOSpreadsheetAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CTO Spreadsheet Specialist",
            goal="Generate professional, data-rich technical Excel workbooks for RailVision's engineering and infrastructure data.",
            backstory=(
                "You are the technical data lead in the Office of the CTO. You transform structured "
                "engineering logs, infrastructure costs, and R&D budgets into clean, professional "
                "Excel workbooks that provide the data-driven foundation for technical decisions."
            ),
            tasks=[
                TaskConfig(
                    description=CTO_SPREADSHEET_PROMPT,
                    expected_output="A professionally generated Excel (.xlsx) file with technical data and a download link."
                )
            ],
        )
        tools = self.tools_provider.get_tools([
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
            ]) if self.tools_provider else []

        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk
