from typing import AsyncGenerator

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class CSODiagnosticsAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService):
        self.llm_provider = llm_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Diagnostics Agent",
            goal="Diagnose issues from logs, traces, and errors and suggest fixes",
            backstory=(
                "You analyze error logs, stack traces, and configuration to pinpoint root causes, "
                "assess impact, and recommend corrective actions with code references."
            ),
            tasks=[
                TaskConfig(
                    description=diagnostics_task_prompt,
                    expected_output=(
                        "Markdown diagnostics report outlining cause analysis, impacted areas, and step-by-step fixes"
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


diagnostics_task_prompt = (
    "Given logs, traces, and error messages, identify likely root causes, configuration issues, "
    "and code-level faults. Map findings to files and modules, suggest precise remediation steps, "
    "and include any needed changes to settings or provider configuration."
)

