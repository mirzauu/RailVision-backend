from typing import AsyncGenerator

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class CSOExplanationAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService):
        self.llm_provider = llm_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Explanation Agent",
            goal="Explain the repository structure, architecture, and key flows",
            backstory=(
                "You are a highly capable explanation agent. You analyze repository structure, "
                "identify important modules, and explain how components interact. You provide clear, "
                "actionable insights, with references and concise summaries suitable for a Chief Security Officer."
            ),
            tasks=[
                TaskConfig(
                    description=explanation_task_prompt,
                    expected_output=(
                        "Markdown explanation of the codebase focused on structure, architecture, and key flows "
                        "with relevant references and file paths"
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


explanation_task_prompt = (
    "Analyze the repository to explain its structure and architecture."
    " Focus on key modules, their responsibilities, entry points (APIs), data flow, and how services integrate."
    " Provide references to important files and paths."
    " Highlight security-relevant components (auth, settings, provider configuration) and any external dependencies."
)

