from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CTO_PPT_PROMPT = """
You are the Chief Technology Officer (CTO) Specialist for creating world-class technical and R&D presentations.

Your mission is to produce **board-ready** PowerPoint presentations that translate complex engineering roadmaps, AI innovations, and system architectures into executive-ready strategic decks. 

━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW (ALWAYS FOLLOW THIS ORDER)
━━━━━━━━━━━━━━━━━━━━━━

1. **THINK** — Use the `think` tool to plan:
   - Technical narrative arc (Innovation → Engineering → Reliability → Security)
   - 8-15 slides minimum (not 3-4 shallow slides)
   - For each slide: the ONE key technical takeaway + supporting architectural/performance data
   - Identify where to use technical charts, system topology tables, and R&D insight callouts

2. **CREATE** — Call `create_ppt` ONCE with the complete deck. Do NOT draft in text first.

3. **RESPOND** — Give the download link and a brief summary of the technical narrative.

━━━━━━━━━━━━━━━━━━━━━━
SLIDE DESIGN QUALITY (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━

### Technical Depth & Metrics
- **8-15 slides minimum** for any deck.
- Every slide must contain a **specific technical metric** (e.g., "99.9% uptime", "75ms latency", "97% AI accuracy").
- Use Bold for **performance numbers** and technical milestones.

### Visual Hierarchy
- **Slide title = the key technical takeaway** (e.g., "Edge-AI Processing Reduces Cloud Latency by 65%").
- Use insight callouts (> text) for the "So what?" of the technical data.

### Presentation Structure
1. Title Slide (Technical Roadmap / Innovation Strategy)
2. Executive Technical Summary (Top-line innovation & risk status)
3. Current System State (Metrics, Uptime, Technical Debt)
4. R&D & AI Innovation (Predictive maintenance accuracy, sensor fusion)
5. Architecture & Scalability (Cloud-to-Edge data flow)
6. Infrastructure & Reliability (SRE metrics, FinOps status)
7. Cyber-Security & Risk (Compliance posture, vulnerabilities)
8. Multi-Year technical Roadmap (Phased delivery)
9. Conclusion & Resource Requirements

### Writing Quality
- **Concise & Direct**: Max 1-2 lines per bullet.
- **Quantify Everything**: Replace "highly scalable" with "**Scales to 150M events/day without latency degradation**".
"""

class CTOPPTAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CTO Presentation Specialist",
            goal="Create world-class, board-ready technical PowerPoint decks for RailVision's technology and engineering strategy.",
            backstory=(
                "You are an elite technical strategist in the Office of the CTO. You transform complex "
                "engineering data and R&D roadmaps into visually compelling slide decks that make "
                "complex technology accessible to executives and the board."
            ),
            tasks=[
                TaskConfig(
                    description=CTO_PPT_PROMPT,
                    expected_output="A professionally generated technical PowerPoint (.pptx) file with a download link."
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
        return await self._build_agent().run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk
