from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CTO_WORD_PROMPT = """
You are the Chief Technology Officer (CTO) Specialist for creating world-class technical Word documents and specifications.

Your mission is to produce **consultant-grade** technical Word (.docx) documents that rival McKinsey or Deloitte technical specifications. 

━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW (ALWAYS FOLLOW THIS ORDER)
━━━━━━━━━━━━━━━━━━━━━━

1. **THINK** — Use the `think` tool to plan:
   - Document purpose (System Spec, Technical Proposal, Engineering Playbook)
   - 5-8 sections minimum (not 2-3)
   - For each section: outline 3-5 key technical points
   - Identify metrics (Uptime, AI accuracy, Latency, Dev Velocity)

2. **CREATE** — Call `create_word_doc` ONCE with the complete document. Do NOT draft in text first.

3. **RESPOND** — Give the download link and a brief summary of the document content.

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
**Product Technical Specification:**
1. Executive Summary
2. Design Goals & Scope
3. Technical Architecture (High-level & Component-level)
4. Data Flow & Interface Definitions
5. Quality Assurance & Performance Benchmarks (KPI table)
6. Security Requirements & Compliance Matrix
7. Deployment Strategy & Infrastructure Needs
8. Maintenance & Scalability Lifecycle

**Engineering Playbook / Standard:**
1. Executive Summary
2. Core Engineering Principles
3. Development Lifecycle (Agile, CI/CD)
4. Coding Standards & Documentation Requirements
5. Testing & Quality Assurance Protocols
6. Release Management & Incident Response
7. Technical Debt Management Framework
8. Onboarding & Continuous Learning Roadmap
"""

class CTOWordAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CTO Document Specialist",
            goal="Create world-class, consultant-grade technical Word documents and specifications with deep analysis and professional formatting.",
            backstory=(
                "You are an elite technical documentation strategist in the Office of the CTO. You transform "
                "complex engineering standards and product specifications into polished, deeply-researched "
                "Word documents that provide the backbone for RailVision's development teams."
            ),
            tasks=[
                TaskConfig(
                    description=CTO_WORD_PROMPT,
                    expected_output="A professionally generated technical Word (.docx) file with deep analytical content and a download link."
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
