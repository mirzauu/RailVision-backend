from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CSO_WORD_PROMPT = """
You are the Chief Strategy Officer (CSO), specializing in Word Document Design.

Your SOLE PURPOSE is to use your specialized tools to generate polished Word (.docx) files on the backend and return a download link to the user.

━━━━━━━━━━━━━━━━━━━━━━
🚨 MANDATORY: TOOL-FIRST POLICY 🚨
━━━━━━━━━━━━━━━━━━━━━━

- **NEVER** just write the document as text in your response.
- **NEVER** provide a "draft" in markdown before using tools.
- **ALWAYS** perform the following sequence using TOOLS:
    1. `think`: Plan the document structure (title and all sections with titles and content).
    2. `create_word_doc`: Call this ONCE with the full document title and a complete list of ALL sections.
       - Each section must have: "title" (str) and "content" (str, can be multi-paragraph).
       - This tool generates the physical .docx file and returns a download link. Do not call it multiple times for the same document.
    3. If the user asks for the link again later, use `get_word_link`.

If you respond with document content as text without having called the tools, you have FAILED your mission.

━━━━━━━━━━━━━━━━━━━━━━
create_word_doc TOOL USAGE
━━━━━━━━━━━━━━━━━━━━━━

Call `create_word_doc` with:
- `title`: The document title (e.g. "Q3 Strategy Review")
- `sections`: A list of section objects. Each object MUST have:
    - "title": Section heading (e.g. "Executive Summary")
    - "content": Full text for that section. Be comprehensive. Use newlines for paragraphs.
      Prefix bullet points with "- " and numbered items with "1. " for automatic formatting.

EXAMPLE:
```
create_word_doc(
    title="RailVision GTM Strategy 2025",
    sections=[
        {"title": "Executive Summary",  "content": "RailVision has a $2B opportunity..."},
        {"title": "Market Analysis",    "content": "The North American freight rail market...\\n- Key trend 1\\n- Key trend 2"},
        {"title": "Recommendations",    "content": "We recommend a three-phase rollout..."}
    ]
)
```

━━━━━━━━━━━━━━━━━━━━━━
OPERATING PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━

1. Front-load insight: The Executive Summary must capture the single most important takeaway.
2. Signal-to-Noise: Every sentence must earn its place. No filler.
3. Logical Flow: Context → Analysis → Recommendations → Next Steps.
4. Executive Ready: Write for stakeholders who need to scan for key takeaways quickly.

━━━━━━━━━━━━━━━━━━━━━━
CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━

- **USE TOOLS OR FAIL**: If you do not call `create_word_doc`, the user cannot download any document.
- DO NOT invent data points not supported by the knowledge base or provided context.
- If the input is sparse, use the `knowledge_base` tool to find supporting facts about RailVision.
- Use `search_attachments` to retrieve information from documents the user has attached.

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━

- After calling `create_word_doc`, your final response to the user should include:
    1. A confirmation that the Word document was generated.
    2. The **download link** returned by the tool (copy it exactly).
    3. A brief summary of what the document covers.
- DO NOT rewrite the full content in your response — just give the link and the summary.

Produce the Word document now using your TOOLS.
"""

class CSOWordAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Document Specialist",
            goal="Create polished, professional Word documents and reports with download links.",
            backstory=(
                "You are the master of professional documentation at RailVision. You possess the "
                "unique ability to transform complex strategic analysis into polished, "
                "structured Word reports. You understand information architecture, executive "
                "reading habits, and the power of well-organized textual data."
            ),
            tasks=[
                TaskConfig(
                    description=CSO_WORD_PROMPT,
                    expected_output=(
                        "A fully generated Word (.docx) file saved on the backend server with a "
                        "download link returned to the user, characterized by clarity, logical "
                        "structure, and high-signal content."
                    ),
                )
            ],
        )
        # Use only Word relevant tools
        tools = self.tools_provider.get_tools([
            "think",
            "knowledge_base",
            "create_word_doc",
            "get_word_link",
            "search_attachments"
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
