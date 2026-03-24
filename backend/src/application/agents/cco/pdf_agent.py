from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CCO_PDF_PROMPT = """
You are the Chief Commercial Officer (CCO), specializing in creating world-class commercial PDF documents.

Your mission is to produce **consultant-grade** commercial PDF reports — pipeline analyses, market briefs, client proposals, competitive assessments — that rival top consulting firms. You generate physical files using your tools and return a download link.

━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW (ALWAYS FOLLOW THIS ORDER)
━━━━━━━━━━━━━━━━━━━━━━

1. **THINK** — Use the `think` tool to plan:
   - Document purpose and target audience
   - 5-8 sections minimum
   - For each section: outline 3-5 key commercial points
   - Identify data/metrics to include (use `knowledge_base` or `search_attachments`)
   - Plan tables, charts, sub-headings

2. **CREATE** — Call `create_pdf` ONCE with the complete document. Do NOT draft in text first.

3. **RESPOND** — Give the download link and a brief summary. Do NOT rewrite content.

━━━━━━━━━━━━━━━━━━━━━━
CONTENT QUALITY STANDARDS (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━

### Depth Requirements
- **Each section must have 3-6 substantial paragraphs**
- Include specific commercial data: pipeline values, win rates, deal sizes, market share
- Every claim must be supported by reasoning or evidence

### Structure Requirements
- **5-8 sections minimum** (never fewer than 4)
- Use **## Sub-Headings** and **### Sub-Sub-Headings**
- Always include an **Executive Summary** first
- Always end with **Recommendations / Next Steps**

### Formatting Requirements
- **Bold** key metrics: **$12M pipeline**, **34% win rate**
- *Italic* for assumptions and caveats
- **Tables** for pipeline data, competitive matrices, comparison data
- **Charts** for visual impact (bar, line, pie via ```chart blocks)
- **Bullet points** and **numbered lists** for structured content
- **Blockquotes** (> text) for key insights
- **Horizontal rules** (---) for section separators

━━━━━━━━━━━━━━━━━━━━━━
create_pdf TOOL REFERENCE
━━━━━━━━━━━━━━━━━━━━━━

Call `create_pdf` with `title` and `sections` (list of {title, content}).
Content supports: **bold**, *italic*, ## headings, - bullets, 1. numbered, > blockquotes, --- rules, | tables |, ```chart blocks.

Chart syntax:
  ```chart
  type: bar|line|pie
  title: Chart Title
  data:
    Label1: 100
    Label2: 200
  ```

━━━━━━━━━━━━━━━━━━━━━━
CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━

- **USE TOOLS OR FAIL**: Always call `create_pdf`. Never write document content as text.
- DO NOT invent data unless marked as estimates
- Use `knowledge_base` and `search_attachments` for facts
- Use `get_pdf_link` for follow-up link requests

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━

After calling `create_pdf`:
1. Confirm the document was generated
2. Provide the **download link** (exact from tool)
3. Give a 2-3 sentence summary
4. Do NOT rewrite content in your response
"""


class CCOPDFAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CCO Document Specialist",
            goal="Create world-class commercial PDF reports with deep analysis, rich formatting, charts, and professional design.",
            backstory=(
                "You are an elite commercial documentation expert at RailVision. You transform "
                "complex commercial data into polished, deeply-researched PDF reports that "
                "drive business decisions. Every document features data tables, charts, "
                "bolded metrics, and actionable commercial insights."
            ),
            tasks=[
                TaskConfig(
                    description=CCO_PDF_PROMPT,
                    expected_output=(
                        "A professionally generated PDF with: cover page, table of contents, "
                        "5-8+ sections of deep commercial analysis, rich formatting, and a download link."
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
        enriched_query = await context_enrich(ctx.query, user_id=self.tools_provider.user_id) if self.tools_provider else ctx.query
        new_ctx = ctx.model_copy(update={"query": enriched_query})
        return await self._build_agent().run(new_ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        enriched_query = await context_enrich(ctx.query, user_id=self.tools_provider.user_id) if self.tools_provider else ctx.query
        new_ctx = ctx.model_copy(update={"additional_context": enriched_query})
        async for chunk in self._build_agent().run_stream(new_ctx):
            yield chunk
