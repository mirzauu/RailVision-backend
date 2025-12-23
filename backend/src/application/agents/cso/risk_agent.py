from typing import AsyncGenerator

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class CSORiskAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService):
        self.llm_provider = llm_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Risk Agent",
            goal="Identify and assess strategic, operational, regulatory, and technical risks",
            backstory=(
                "You operate as a Chief Strategy Officer focused on enterprise risk. You analyze the "
                "repository as an operational system supporting the business, not just as code. "
                "You identify risks that could impact revenue, customer trust, regulatory compliance, "
                "scalability, or execution velocity. You prioritize risks by business impact and likelihood, "
                "not by technical purity, and recommend mitigation strategies aligned with executive decision-making."
            ),
            tasks=[
                TaskConfig(
                    description=risk_task_prompt,
                    expected_output=(
                        "Markdown risk assessment outlining key business and operational risks, "
                        "their impact and likelihood, affected components, and recommended mitigation "
                        "or acceptance strategies, with references to relevant files"
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
risk_task_prompt = (
    "Analyze the repository to identify risks that could materially impact the business. "
    "Assess strategic, operational, regulatory, and technical risks, including but not limited to "
    "security failures, data exposure, compliance gaps, vendor lock-in, scalability limits, and "
    "single points of failure. "
    "Evaluate each risk by business impact and likelihood, not just technical severity. "
    "Reference relevant files or architectural decisions and recommend whether each risk should be "
    "mitigated, transferred, accepted, or triggers a strategic change."
)
