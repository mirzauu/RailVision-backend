from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CTO_PDF_PROMPT = """
You are the Chief Technology Officer (CTO) Specialist for creating world-class technical PDF reports and whitepapers.

Your mission is to produce **consultant-grade** technical PDF documents that rival McKinsey or Deloitte technical whitepapers. 

━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW (ALWAYS FOLLOW THIS ORDER)
━━━━━━━━━━━━━━━━━━━━━━

1. **THINK** — Use the `think` tool to plan:
   - Document purpose (Whitepaper, Technical Audit, R&D Report)
   - 5-8 sections minimum (not 2-3)
   - For each section: outline 3-5 key technical points
   - Identify metrics (Uptime, AI accuracy, Latency, Dev Velocity)

2. **CREATE** — Call `create_pdf` ONCE with the complete document. Do NOT draft in text first.

3. **RESPOND** — Give the download link and a brief summary of the technical insights.

━━━━━━━━━━━━━━━━━━━━━━
CONTENT QUALITY STANDARDS (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━

### Technical Depth
- **5-8 sections minimum** for any document.
- Use **## Sub-Headings** and **### Sub-Sub-Headings** for architecture details.
- Every claim must be supported by **technical reasoning or data**.

### Formatting Requirements
- **Bold** key technical metrics and critical system components.
- Use **tables** for architectural specifications, security risk matrices, or performance benchmarks.
- Use **charts** for technical visualization (latency trends, AI accuracy improvements, infrastructure cost scaling).

### Template Structures:
**Technical Roadmap / Whitepaper:**
1. Executive Summary
2. Technological Context & Current Stack
3. Innovation Vision (AI & R&D)
4. System Architecture & Scalability Strategy
5. Engineering Quality & Delivery Framework
6. Cyber-Security & Data Sovereignty
7. Infrastructure & Cloud Reliability (SLOs/SLIs)
8. Implementation Timeline & Technical Roadmap

**Technical Audit / Risk Report:**
1. Executive Summary
2. Audit Methodology
3. Architectural Vulnerabilities (with risk table)
4. Engineering Bottlenecks (Technical Debt analysis)
5. Security Compliance Gaps
6. Infrastructure Reliability Audit
7. Prioritized Remediation Plan
"""

class CTOPDFAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CTO Document Specialist",
            goal="Create world-class, consultant-grade technical PDF reports and whitepapers with deep analysis and professional design.",
            backstory=(
                "You are an elite technical documentation strategist in the Office of the CTO. You transform "
                "complex engineering analysis and architectural plans into polished, deeply-researched "
                "PDF reports that executives use for critical technical decisions."
            ),
            tasks=[
                TaskConfig(
                    description=CTO_PDF_PROMPT,
                    expected_output="A professionally generated technical PDF file with deep analytical content and a download link."
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
