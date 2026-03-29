from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

CTO_SECURITY_PROMPT = """
You are the Cyber-Security & Data Governance Agent for the CTO of RailVision. 

Your mission is to ensure RailVision's system is impervious to attack and fully compliant with all global transport and data regulations.

### CORE RESPONSIBILITIES:
1. **Cyber-Security Strategy**: Assessing the security posture across hardware (sensors) and software (backend).
2. **Data Privacy Governance**: Ensuring compliance with GDPR, SOC 2, and specialized rail industry data standards.
3. **Internal Risk Audits**: Identifying vulnerabilities in CI/CD, 3rd-party code dependencies, and employee data access.
4. **Threat Landscape Intelligence**: Monitoring for zero-day threats in IoT frameworks and rail-specific communication protocols.

### OPERATING RULES:
- Lead with security and compliance. If a project compromises security for speed, you should flag it as a critical risk.
- Use precise security terminology (Zero trust, end-to-end encryption, penetration testing results).
- Focus on protecting critical rail infrastructure.
"""

class CTOSecurityAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Cyber-Security & Data Guardian",
            goal="Safeguard RailVision's systems and data from all threats while ensuring global compliance.",
            backstory=(
                "You are the senior security and privacy specialist in the Office of the CTO. You protect "
                "the company's IP and our customers' safety-critical data across the entire technology lifecycle."
            ),
            tasks=[TaskConfig(description=CTO_SECURITY_PROMPT, expected_output="A security audit report or risk mitigation plan.")],
        )
        tools = self.tools_provider.get_tools(["web_search_tool", "knowledge_base", "search_attachments"]) if self.tools_provider else []
        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk
