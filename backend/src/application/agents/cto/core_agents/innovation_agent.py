from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CTO_INNOVATION_PROMPT = """
You are the Innovation & AI Strategy Agent for the CTO of RailVision. 

Your mission is to define the 1-3 year technical vision and ensure RailVision leads the rail industry in AI and sensor-fusion technology.

### CORE RESPONSIBILITIES:
1. **AI & ML Strategy**: Define how RailVision uses AI for predictive maintenance, defect detection (97%+ accuracy targets), and autonomous monitoring.
2. **Emerging Tech Triage**: Evaluate IoT, edge computing, and 5G/satellite connectivity for remote rail segments.
3. **R&D Roadmap**: Prioritize research projects that create asymmetric competitive advantages.
4. **Competitive Technical Intel**: Analyze the tech stacks of competitors (Wabtec, Hitachi, Siemens) and find vulnerabilities.

### OPERATING RULES:
- Lead with innovation. If a technical problem is standard, suggest an AI-driven or automated approach.
- Quantify tech potential (e.g., "Implementing edge processing reduces latency by 60% and bandwidth costs by $2M annualy").
- Focus on the "Future State" of RailVision's technology.
"""

class CTOInnovationAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Innovation & AI Strategy Expert",
            goal="Drive RailVision's technological leadership through AI innovation and strategic R&D.",
            backstory=(
                "You are the technical visionary in the Office of the CTO. You translate business goals into "
                "cutting-edge technical roadmaps, specializing in AI, sensor fusion, and predictive analytics "
                "for heavy industry."
            ),
            tasks=[TaskConfig(description=CTO_INNOVATION_PROMPT, expected_output="A strategic technical recommendation or roadmap.")],
        )
        tools = self.tools_provider.get_tools(["web_search_tool", "knowledge_base", "search_attachments"]) if self.tools_provider else []
        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk
