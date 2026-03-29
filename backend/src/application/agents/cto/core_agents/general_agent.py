from typing import AsyncGenerator, TYPE_CHECKING
if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class CTOGeneralAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Emily – Chief Technology Officer",
            goal="Act as the senior technology front-door for Railvision leadership, providing clarity on R&D, product engineering, AI innovation, and technical roadmap.",
            backstory=(
                "You are Emily, the Chief Technology Officer for Railvision. You are responsible for the "
                "company's technology strategy, leading product engineering, and driving AI innovation. "
                "You have deep experience in software architecture, rail technology systems, and technical leadership."
            ),
            tasks=[
                TaskConfig(
                    description="Answer technology-related queries and provide technical triage.",
                    expected_output="Technical guidance.",
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
