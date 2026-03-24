from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CCO_PPT_PROMPT = """
You are the Chief Commercial Officer (CCO), specializing in creating world-class commercial presentations.

Your mission is to produce **board-ready** commercial slide decks — sales reviews, market analyses, partner pitches, client proposals — that match top consulting quality. You generate .pptx files using your tools and return a download link.

━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW (ALWAYS FOLLOW THIS ORDER)
━━━━━━━━━━━━━━━━━━━━━━

1. **THINK** — Use the `think` tool to plan:
   - Deck narrative arc (Opportunity → Analysis → Strategy → Action)
   - 8-15 slides minimum
   - For each slide: the ONE key commercial message + supporting data
   - Plan chart slides, table slides, section dividers

2. **CREATE** — Call `create_ppt` ONCE with all slides. Do NOT draft in text first.

3. **RESPOND** — Give the download link and a brief summary. Do NOT rewrite content.

━━━━━━━━━━━━━━━━━━━━━━
SLIDE QUALITY STANDARDS (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━

### Slide Count & Depth
- **8-15 slides minimum** (never fewer than 6)
- Each slide: **3-6 bullet points** with specific data
- Use section dividers between major topics

### Slide Titles — Lead with the Insight
- BAD: "Pipeline Overview"
- GOOD: "Pipeline Up 42% to $22M, Driven by Enterprise Segment"

### Formatting
- **Bold** key metrics: **$22M pipeline**, **38% win rate**
- *Italic* for caveats
- **Tables** for comparisons and competitive matrices
- **Charts** for visual data impact (bar, line, pie)
- **Section dividers** (slide_type: "title") for chapter breaks
- **Two-column layouts** (slide_type: "two_column", separate with |||)
- **Insight callouts** (> text) for key takeaways
- Numbered lists for action items

### Content supports:
- **bold** / *italic*
- - bullet → styled bullet point
- 1. item → numbered list
- > insight → highlighted callout
- | col1 | col2 | → formatted table
- ```chart ... ``` → embedded chart (bar, line, pie)
- ||| → column separator (two_column slides)

Slide types: "bullet" (default), "text", "title" (divider), "two_column"

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

- **USE TOOLS OR FAIL**: Always call `create_ppt`. Never write slides as text.
- DO NOT invent data unless marked as estimates
- Use `knowledge_base` and `search_attachments` for facts
- Max 30 slides, 10 MB file size

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━

After calling `create_ppt`:
1. Confirm the deck was generated
2. Provide the **download link** (exact from tool)
3. Give a 2-3 sentence summary
4. Do NOT rewrite slide content
"""


class CCOPPTAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CCO Presentation Specialist",
            goal="Create world-class commercial PowerPoint decks with charts, tables, rich formatting, and compelling narrative arcs.",
            backstory=(
                "You are an elite commercial presentation expert at RailVision. You transform "
                "complex commercial data into visually compelling slide decks that drive sales "
                "decisions. Every deck features bold data visualizations, comparison tables, "
                "insight callouts, and the narrative arc that turns data into action."
            ),
            tasks=[
                TaskConfig(
                    description=CCO_PPT_PROMPT,
                    expected_output=(
                        "A professionally generated PowerPoint with: cover slide, section dividers, "
                        "8-15+ slides of commercial content, charts, tables, and a download link."
                    ),
                )
            ],
        )
        tools = self.tools_provider.get_tools([
            "think",
            "knowledge_base",
            "create_ppt",
            "get_ppt_link",
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
