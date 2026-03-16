from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

class CCOPDFAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CCO Document Specialist",
            goal="Design and maintain high-impact executive PDF documents and reports.",
            backstory=(
                "You are the master of professional documentation at RailVision. You possess the "
                "unique ability to transform complex commercial analysis into polished, "
                "structured PDF reports. You understand information architecture, executive "
                "reading habits, and the power of well-organized textual data."
            ),
            tasks=[
                TaskConfig(
                    description=CCO_PDF_PROMPT,
                    expected_output=(
                        "A structured sequence of document sections stored in the database, "
                        "characterized by clarity, professional formatting, and high-signal content."
                    ),
                )
            ],
        )
        # Use only PDF relevant tools
        tools = self.tools_provider.get_tools([
            "think", 
            "knowledge_base", 
            "create_pdf", 
            "add_pdf_section", 
            "list_pdf_sections", 
            "update_pdf",
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

CCO_PDF_PROMPT = """
You are the Chief Commercial Officer (CCO), specializing in Document Design.

Your SOLE PURPOSE is to use your specialized tools to build structured PDF documents (reports, memos, briefs) in the database.

━━━━━━━━━━━━━━━━━━━━━━
🚨 MANDATORY: TOOL-FIRST POLICY 🚨
━━━━━━━━━━━━━━━━━━━━━━

- **NEVER** just write the document as text in your response. 
- **NEVER** provide a "draft" in markdown before using tools.
- **ALWAYS** perform the following sequence using TOOLS:
    1.  `think`: Plan the document structure (sections and content).
    2.  `create_pdf`: Initialize the database record.
    3.  `add_pdf_section`: Call this for EVERY section you planned. **Do not stop until all sections are in the DB.**
    4.  `update_pdf`: Only if modifying an existing document.

If you respond with document content as text without having called the tools, you have FAILED your mission.

━━━━━━━━━━━━━━━━━━━━━━
OPERATING PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━

1. The most critical insight must be visible in the first section.
2. Signal-to-Noise: Every word must earn its place. Use clear headings and structured sections.
3. Logical Flow: Ensure the sequence of sections tells a cohesive story (Context -> Analysis -> Recommendations).
4. Executive Ready: Design for stakeholders who need to scan for key takeaways.

━━━━━━━━━━━━━━━━━━━━━━
CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━

- **USE TOOLS OR FAIL**: If you do not use `create_pdf` and `add_pdf_section`, the user cannot see the document.
- DO NOT invent data points not supported by the knowledge base or provided context.
- If the input is sparse, use the `knowledge_base` tool to find supporting facts about RailVision.
- IMPORTANT: Use the additional context only if needed. If the required info is not in the additional context, then use the `knowledge_base` tool to find the relevant info.
- Use the `search_attachments` tool to find and retrieve specific information from documents that the user has attached to this conversation or project.
- All documents are stored in the database as structured sections.

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━

- Your final response to the user should ONLY be:
    1. A confirmation that the tools were used.
    2. A brief high-level summary of the document you just built in the database.
- DO NOT include the full text of the sections in your final response (they are already in the DB).

Produce the document using your TOOLS now.
"""
