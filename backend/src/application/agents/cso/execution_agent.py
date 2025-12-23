from typing import AsyncGenerator

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class CSOExecutionAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService):
        self.llm_provider = llm_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Execution Agent",
            goal="Evaluate execution readiness, delivery risk, and operational bottlenecks",
            backstory=(
                "You operate as a Chief Strategy Officer focused on execution. You assess whether the "
                "current system, processes, and technical foundations can reliably deliver the business "
                "strategy. You identify execution bottlenecks, ownership gaps, fragile dependencies, "
                "and decisions that slow down iteration, increase cost, or create delivery risk. "
                "You translate technical signals into execution-level insights for leadership."
            ),
            tasks=[
                TaskConfig(
                    description=execution_task_prompt,
                    expected_output=(
                        "Markdown execution assessment identifying delivery risks, bottlenecks, "
                        "organizational or technical constraints, and prioritized actions to improve "
                        "speed, reliability, and execution confidence, with references to relevant components"
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

execution_task_prompt = (
    "Analyze the repository and operational signals to evaluate execution readiness. "
    "Identify technical and organizational bottlenecks, brittle dependencies, unclear ownership, "
    "poor separation of concerns, and decisions that slow delivery or increase operational load. "
    "Assess whether the current implementation supports rapid iteration, reliable deployment, "
    "and scaling with team growth. "
    "Reference relevant files or architectural patterns and recommend prioritized actions "
    "to unblock execution and improve delivery confidence."
)
