from typing import AsyncGenerator, TYPE_CHECKING
if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class PartnerChannelAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Partner & Channel Revenue Manager",
            goal="Develop and manage channel partnerships, alliances, and indirect revenue streams.",
            backstory=(
                "You are a Partner & Channel Revenue Manager working under Gabrial, the Chief Revenue Officer for Railvision. "
                "You focus on identifying, onboarding, and enabling strategic partners to sell Railvision products. You manage "
                "partner performance, channel conflict, and joint go-to-market initiatives."
            ),
            tasks=[
                TaskConfig(
                    description="Assess channel partner performance and identify new alliance opportunities.",
                    expected_output="Channel strategy review and partnership proposals.",
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
