from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from .base import ChatAgent, ChatAgentResponse, ChatContext, AgentConfig
from .pydantic_agent import PydanticChatAgent
from .crewai_agent import CrewAIChatAgent


class ExecuterAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, config: AgentConfig, framework: str = "pydantic"):
        self.framework = framework.lower()
        self.pydantic_agent: Optional[PydanticChatAgent] = None
        self.crewai_agent: Optional[CrewAIChatAgent] = None
        if self.framework == "pydantic":
            self.pydantic_agent = PydanticChatAgent(llm_provider, config)
        elif self.framework == "crewai":
            self.crewai_agent = CrewAIChatAgent(config)
        else:
            self.pydantic_agent = PydanticChatAgent(llm_provider, config)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        if self.pydantic_agent:
            return await self.pydantic_agent.run(ctx)
        if self.crewai_agent:
            return await self.crewai_agent.run(ctx)
        return ChatAgentResponse(response="", tool_calls=[], citations=[])

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        if self.pydantic_agent:
            async for chunk in self.pydantic_agent.run_stream(ctx):
                yield chunk
            return
        if self.crewai_agent:
            async for chunk in self.crewai_agent.run_stream(ctx):
                yield chunk
            return

