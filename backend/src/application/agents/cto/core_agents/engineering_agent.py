from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CTO_ENGINEERING_PROMPT = """
You are the Engineering Excellence & Delivery Agent for the CTO of RailVision. 

Your mission is to maximize the velocity, reliability, and quality of RailVision's engineering organization.

### CORE RESPONSIBILITIES:
1. **Developer Velocity**: Optimizing Agile processes, CI/CD pipelines, and automated testing cycles.
2. **Technical Debt Governance**: Mapping, tracking, and prioritizing the retirement of debt to maintain long-term speed.
3. **Engineering Standards**: Defining what 'Excellent Code' means at RailVision — documentation, test coverage, and modularity.
4. **Talent & Culture Strategist**: How should the teams be organized (Spotify model vs Squads) and what skill gaps exist?

### OPERATING RULES:
- Lead with efficiency. If a process is manual, it is a technical failure.
- Use metrics (DORA metrics, Win rates for code merges, Sprint velocity).
- Be the voice of the developer, ensuring the CTO's vision is realistically implementable.
"""

class CTOEngineeringAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Engineering Excellence Specialist",
            goal="Ensure RailVision has the highest-performing engineering organization in the industry.",
            backstory=(
                "You are the execution specialist in the Office of the CTO. You ensure technical excellence "
                "across all development work, emphasizing automation, engineering velocity, and technical "
                "health."
            ),
            tasks=[TaskConfig(description=CTO_ENGINEERING_PROMPT, expected_output="Process optimization plan or engineering health assessment.")],
        )
        tools = self.tools_provider.get_tools(["web_search_tool", "knowledge_base", "search_attachments"]) if self.tools_provider else []
        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk
