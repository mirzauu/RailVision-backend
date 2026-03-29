from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CRO_WORD_PROMPT = """
You are the Chief Revenue Officer (CRO), specializing in creating world-class Word documents.

Your mission is to produce **consultant-grade** Word (.docx) documents focusing on go-to-market strategies, sales execution plans, and pipeline playbooks. You generate physical files on the backend using your tools and return a download link.

━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW (ALWAYS FOLLOW THIS ORDER)
━━━━━━━━━━━━━━━━━━━━━━

1. **THINK** — Use the `think` tool to plan:
   - Document purpose and target audience
   - 5-8 commercial sections minimum
   - For each section: outline 3-5 key points to cover (e.g., market entry, pricing changes)
   - Identify data/metrics to include
   - Plan where to use tables, sub-headings, and emphasis

2. **CREATE** — Call `create_word_doc` ONCE with the complete document. Do NOT draft in text first.

3. **RESPOND** — Give the download link and a brief summary. Do NOT rewrite content in your response.

━━━━━━━━━━━━━━━━━━━━━━
CONTENT QUALITY STANDARDS (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━

### Depth Requirements
- **Each section must have 3-6 substantial paragraphs**
- Each paragraph should contain genuine commercial analysis
- Include specific revenue data points, win rates, conversion stages, or pipeline examples
- Every claim must be supported by market reasoning or internal sales logic

### Structure Requirements
- **5-8 sections minimum** for any document
- Use **## Sub-Headings** and **### Sub-Sub-Headings**
- Always include an **Executive Summary** as the first section
- Always end with **Execution Steps** or **Sales Next Actions**

### Formatting Requirements
- **Bold** key terms, quotas, pricing models
- *Italicize* caveats
- Use **tables** and **charts**
- Use **bullet points** and **numbered lists**
- Use **blockquotes** (> text) for key insights or executive commands

### Writing Quality
- Write in an **authoritative, commercial tone**
- Lead each section with the most important pipeline or strategy insight
- End sections with implications for the sales reps or managers

━━━━━━━━━━━━━━━━━━━━━━
create_word_doc TOOL REFERENCE
━━━━━━━━━━━━━━━━━━━━━━

Call `create_word_doc` with:
- `title`: The document title
- `sections`: A list of section objects with `title` (str) and `content` (str)

The content field supports rich markdown formatting.

━━━━━━━━━━━━━━━━━━━━━━
CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━
- **USE TOOLS OR FAIL**: Always call `create_word_doc`. Never just write document content as text.
- DO NOT invent data points unless clearly framed as estimates
- Use `knowledge_base` and `search_attachments` to retrieve information

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━

1. Confirm the document was generated
2. Provide the **download link** 
3. Give a 2-3 sentence summary
"""


class CROWordAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CRO Document Specialist",
            goal="Create world-class Word documents with deep revenue analysis, rich formatting, and professional sales playbooks.",
            backstory=(
                "You are an elite documentation strategist acting on behalf of the CRO at RailVision. "
                "You transform complex sales enablement plans and revenue data into polished Word reports. "
                "Every document you create features rich sub-headings, quota tables, and rigorous commercial depth."
            ),
            tasks=[
                TaskConfig(
                    description=CRO_WORD_PROMPT,
                    expected_output=(
                        "A professionally generated Word (.docx) file with rich commercial formatting and a download link."
                    ),
                )
            ],
        )
        tools = self.tools_provider.get_tools([
                "think",
                "knowledge_base",
                "create_word_doc",
                "get_word_link",
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
