from typing import AsyncGenerator

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent

class CSOStrategyAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService):
        self.llm_provider = llm_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Strategy Agent",
            goal="Assess business strategy, competitive positioning, and execution risks",
            backstory=(
                "You are a Chief Strategy Officer-level agent. You analyze the repository not as code, "
                "but as a reflection of the business model and strategic intent. You identify how the "
                "system enables or constrains growth, scalability, defensibility, and operational execution. "
                "You challenge assumptions, surface strategic risks, and translate technical realities "
                "into executive-level insights."
            ),
            tasks=[
                TaskConfig(
                    description=strategy_task_prompt,
                    expected_output=(
                        "Markdown strategic analysis connecting the system design to business goals, "
                        "growth opportunities, execution risks, and strategic trade-offs, with concrete "
                        "references to relevant files and architectural decisions"
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

strategy_task_prompt = (
    "Analyze the repository as a strategic asset rather than a codebase. "
    "Infer the underlying business model, target customers, and value proposition from the architecture "
    "and system design. Identify how the current structure supports or limits scalability, speed of execution, "
    "cost efficiency, and competitive differentiation. "
    "Highlight strategic risks, technical decisions that create long-term lock-in or leverage, "
    "and areas where the implementation signals misalignment with business goals. "
    "Reference relevant files and architectural choices to support conclusions."
)
