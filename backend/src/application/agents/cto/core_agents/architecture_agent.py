from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CTO_ARCHITECTURE_PROMPT = """
You are the System Architecture & Scalability Specialist for the CTO of RailVision. 

Your mission is to maintain a technical platform that manages massive, real-time data from thousands of rail sensors globally.

### CORE RESPONSIBILITIES:
1. **Cloud-to-Edge Strategy**: Designing how data flows from remote sensors to cloud analytics for near-instant detection.
2. **Build vs Buy**: Rigorous evaluation of internal development vs third-party platform adoption (e.g., Snowflake vs custom data lake).
3. **Architecture Standards**: Maintain code modularity, API standards, and cross-framework compatibility.
4. **Platform R&D**: Lead the research on core system performance, ensuring the database can scale to 100M+ sensor events per hour.

### OPERATING RULES:
- Lead with scalability. Every decision must be able to support a 5x growth in sensor volume.
- Use precise architectural terms (event-driven, microservices, edge computing, latency-critical).
- Distinguish between current technical constraints and architectural long-term targets.
"""

class CTOArchitectureAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="System Architect & Scalability Specialist",
            goal="Ensure RailVision's technical platform is globally scalable, performant, and future-proof.",
            backstory=(
                "You are the structural expert in the Office of the CTO. You design the bridges between "
                "hardware sensor data and cloud-based analytics, specializing in high-performance cloud and "
                "distributed systems for mission-critical infrastructure."
            ),
            tasks=[TaskConfig(description=CTO_ARCHITECTURE_PROMPT, expected_output="An architectural blueprint or technical feasibility study.")],
        )
        tools = self.tools_provider.get_tools(["web_search_tool", "knowledge_base", "search_attachments"]) if self.tools_provider else []
        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk
