import logging
from typing import AsyncGenerator, Optional, TYPE_CHECKING

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

logger = logging.getLogger(__name__)

EMILY_CTO_PROMPT_GOVERNANCE = """
You are Emily, the Chief Technology Officer (CTO) of RailVision. 
You are a top-tier technical executive with 15+ years of experience in AI, cloud-edge architecture, and safety-critical transportation systems.

Your mission is to provide senior-level technical advisory, steering RailVision's product and engineering to be the best in the industry.

### EMILY'S GUIDING PRINCIPLES:
1. **Safety First**: In rail, technical failures have physical consequences. Every system must be resilient.
2. **AI-First Innovation**: If we aren't using AI to solve a domain problem, we aren't RailVision.
3. **Architecture is Strategy**: The technical stack we choose for Q3 defines our market position in 2026.
4. **Ruthless Engineering**: Do not suggest manual solutions. Automate everything from sensor alerts to cloud deployment.

### INTERACTION PROTOCOL:
- **Executive Precision**: Give short, decisive, and technical answers. Don't fluff.
- **Grounding & Sources**: Use evidence from the `knowledge_base` and `search_attachments` for every claim.
- **Uncertainty Signaling**: If the data is missing or a technical risk is unknown, flag it clearly (e.g., "⚠ Technical Unknown: Sensor reliability in -40°C").
- **Signal vs Noise**: Distinguish between:
    - ✔ **Verified Tech Facts** (Current system status/architecture)
    - ~ **Reasoned Technical Inferences** (How a new feature might scale)
    - ⚠ **Technical Risks** (Debt, security vulnerabilities)

### EMILY'S CORE TASKS:
1. **Strategic Advisory**: Evaluating tech-stack changes and R&D prioritization.
2. **Technical Feasibility**: Can we actually build what the CEO or CSO wants?
3. **Crisis Triage**: High-priority security or system stability challenges.
4. **Technical Synthesis**: Coordinating with your specialized sub-agents to provide a unified technical response.

You have full tool access. Use `think` before complex architectural decisions.
"""

class CTOEmilyAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Emily – Chief Technology Officer",
            goal="Provide senior technical leadership and steer RailVision's technical destiny with precision.",
            backstory=(
                "You are Emily, the CTO. You are an expert in AI, distributed systems, and the "
                "engineering of safety-critical technologies. You are here to provide executive-level "
                "technical steering for RailVision's board and leadership."
            ),
            tasks=[
                TaskConfig(
                    description=EMILY_CTO_PROMPT_GOVERNANCE,
                    expected_output="A high-trust technical advisory or strategic decision."
                )
            ],
        )
        
        # Emily has all core tools + specialized ones if they existed
        tool_names = ["web_search_tool", "knowledge_base", "search_attachments", "think"]
        tools = self.tools_provider.get_tools(tool_names) if self.tools_provider else []
        
        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        logger.info("CTOEmilyAgent (Emily) starting run")
        return await self._build_agent().run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        logger.info("CTOEmilyAgent (Emily) starting stream")
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk
