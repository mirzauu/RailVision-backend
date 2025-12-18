from typing import AsyncGenerator

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class CSOAuditorAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService):
        self.llm_provider = llm_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Auditor Agent",
            goal="Audit repository for security risks and misconfigurations",
            backstory=(
                "You perform thorough security audits focusing on authentication, secrets handling, "
                "configuration management, logging, and external integrations. You identify risks, "
                "prioritize findings, and provide actionable recommendations with references."
            ),
            tasks=[
                TaskConfig(
                    description=auditor_task_prompt,
                    expected_output=(
                        "Markdown audit report with findings, severity, affected files, and remediation steps"
                    ),
                )
            ],
        )
        return PydanticChatAgent(self.llm_provider, agent_config, tools=[])

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


auditor_task_prompt = (
    "Perform a security audit focusing on: auth flows, token issuance and storage, permission checks, "
    "settings and environment variable usage, provider configuration, external HTTP calls, logging of sensitive data, "
    "and error handling. Identify risks, cite files and lines where possible, and propose mitigations with priority."
)
