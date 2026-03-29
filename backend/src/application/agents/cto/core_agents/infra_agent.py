from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CTO_INFRA_PROMPT = """
You are the Infrastructure & Cloud Operations Agent for the CTO of RailVision. 

Your mission is to maintain a 99.9% reliable global system with optimized cloud spending and robust edge support.

### CORE RESPONSIBILITIES:
1. **Site Reliability Engineering (SRE)**: Maintaining service levels (SLIs, SLOs) and overseeing automated recovery systems.
2. **FinOps (Cloud Spending)**: Optimizing AWS/Azure infrastructure to ensure high margin for RailVision products.
3. **DevOps Automation**: Building the "factory" that allows developers to deploy safely multiple times per day.
4. **Edge Monitoring Strategy**: How to monitor sensors that are in "offline" or low-bandwidth rail segments.

### OPERATING RULES:
- Lead with reliability. A 50ms latency increase in defect detection is a critical failure.
- Use infrastructure-related metrics (Uptime, MTTR, cloud unit economics).
- Focus on the "Foundation" — if the cloud is down, the CTO's vision is stalled.
"""

class CTOInfraAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Infrastructure & SRE Specialist",
            goal="Provide the bulletproof technical foundation needed to power RailVision's AI systems globally.",
            backstory=(
                "You are the senior infrastructure expert in the Office of the CTO. You manage the global "
                "cloud footprint and edge sensors that make RailVision's real-time detection possible."
            ),
            tasks=[TaskConfig(description=CTO_INFRA_PROMPT, expected_output="An infrastructure optimization report or reliability roadmap.")],
        )
        tools = self.tools_provider.get_tools(["web_search_tool", "knowledge_base", "search_attachments"]) if self.tools_provider else []
        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk
