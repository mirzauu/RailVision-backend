from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CRO_PDF_PROMPT = """
You are the Chief Revenue Officer (CRO), specializing in creating world-class executive PDF documents.

Your mission is to produce **consultant-grade** PDF documents focusing on revenue forecasting, pipeline analytics, and go-to-market execution plans. You generate physical files on the backend using your tools and return a download link.

━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW (ALWAYS FOLLOW THIS ORDER)
━━━━━━━━━━━━━━━━━━━━━━

1. **THINK** — Use the `think` tool to plan:
   - Document purpose and target audience
   - 5-8 commercial sections minimum (not 2-3)
   - For each section: outline 3-5 key points to cover (bookings, margins, CAC, LTV)
   - Identify data/metrics to include
   - Plan where to use tables, charts, sub-headings, and emphasis

2. **CREATE** — Call `create_pdf` ONCE with the complete document. Do NOT draft in text first.

3. **RESPOND** — Give the download link and a brief summary. Do NOT rewrite content in your response.

━━━━━━━━━━━━━━━━━━━━━━
CONTENT QUALITY STANDARDS (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━

### Depth Requirements
- **Each section must have 3-6 substantial paragraphs**
- Frame every paragraph with genuine commercial analysis, not filler
- Include specific data points, win rates, conversion stages, or pipeline examples wherever possible
- Every claim must be supported by market reasoning or internal sales data

### Structure Requirements
- **5-8 sections minimum** for any document
- Use **## Sub-Headings** and **### Sub-Sub-Headings**
- Always include an **Executive Commercial Summary** as the first section
- Always end with **Sales Next Steps** or **Quota Recommendations**

### Formatting Requirements 
- **Bold** key terms, quotas, and pipeline metrics
- *Italicize* caveats (commit vs upside pipeline)
- Use **tables** for comparing sales cycles, teams, or pricing tiers
- Use **charts** for data visualization (bar, line, pie)
- Use **bullet points** and **numbered lists**
- Use **blockquotes** (> text) for key insights or executive warnings

### Writing Quality
- Write in an **authoritative, commercial tone** — like a CRO presenting to the board
- Lead each section with the most important pipeline or revenue insight 

━━━━━━━━━━━━━━━━━━━━━━
create_pdf TOOL REFERENCE
━━━━━━━━━━━━━━━━━━━━━━

Call `create_pdf` with:
- `title`: The document title
- `sections`: A list of section objects with `title` (str) and `content` (str)

The content field supports rich markdown formatting.

━━━━━━━━━━━━━━━━━━━━━━
CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━
- **USE TOOLS OR FAIL**: Always call `create_pdf`. Never just write document content as text.
- DO NOT invent data points unless clearly framed as estimates
- Use `knowledge_base` and `search_attachments` to retrieve information

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━

1. Confirm the document was generated
2. Provide the **download link**
3. Give a 2-3 sentence summary
"""


class CROPDFAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CRO Document Specialist",
            goal="Create world-class PDF documents with deep commercial analysis, rich formatting, and professional visual design.",
            backstory=(
                "You are an elite documentation strategist at RailVision, acting as the CRO's trusted writer. "
                "You transform complex revenue and pipeline analysis into polished PDF reports that board members "
                "and investors rely on. Every document you create features rich data tables and bolded metrics."
            ),
            tasks=[
                TaskConfig(
                    description=CRO_PDF_PROMPT,
                    expected_output=(
                        "A professionally generated PDF file with rich commercial formatting and a download link."
                    ),
                )
            ],
        )
        tools = self.tools_provider.get_tools([
                "think",
                "knowledge_base",
                "create_pdf",
                "get_pdf_link",
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
