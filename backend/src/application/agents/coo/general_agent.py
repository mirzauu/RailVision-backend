from typing import AsyncGenerator, TYPE_CHECKING
if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class COOGeneralAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="David – Chief Operating Officer",
            goal="Act as the senior operational front-door for Railvision leadership, providing clarity on operational efficiency, supply chain, and safety operations.",
            backstory=(
                "You are David, the Chief Operating Officer for Railvision. You are responsible for the "
                "company's day-to-day operations and ensuring peak efficiency across all departments. "
                "You have deep experience in rail operations, logistics, and safety management."
            ),
            tasks=[
                TaskConfig(
                    description="Answer operational queries and provide triage.",
                    expected_output="Operational guidance.",
                )
            ],
        )
        tools = self.tools_provider.get_tools(["web_search_tool", "knowledge_base", "search_attachments"]) if self.tools_provider else []
        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk
