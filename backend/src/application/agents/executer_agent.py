from typing import AsyncGenerator, Optional
import logging

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import ChatAgent, ChatAgentResponse, ChatContext, AgentConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.infrastructure.agents.crewai_agent import CrewAIChatAgent
from src.application.agents.cso.router_agent import CSORouterAgent

logger = logging.getLogger(__name__)


class ExecuterAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, config: AgentConfig, framework: str = "pydantic"):
        self.framework = framework.lower()
        self.pydantic_agent: Optional[PydanticChatAgent] = None
        self.crewai_agent: Optional[CrewAIChatAgent] = None
        self.router_agent: Optional[CSORouterAgent] = None
        if self.framework == "pydantic":
            self.pydantic_agent = PydanticChatAgent(llm_provider, config)
        elif self.framework == "crewai":
            self.crewai_agent = CrewAIChatAgent(config)
        elif self.framework in {"router", "cso_router", "cso"}:
            self.router_agent = CSORouterAgent(llm_provider)
        else:
            self.pydantic_agent = PydanticChatAgent(llm_provider, config)
        chosen = (
            "router" if self.router_agent else ("pydantic" if self.pydantic_agent else ("crewai" if self.crewai_agent else "none"))
        )
        logger.info("ExecuterAgent initialized using framework '%s' -> '%s'", self.framework, chosen)
        print(f"ExecuterAgent initialized using framework '{self.framework}' -> '{chosen}'")

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        if self.router_agent:
            print("ExecuterAgent delegating to CSORouterAgent")
            logger.info("ExecuterAgent delegating to CSORouterAgent")
            return await self.router_agent.run(ctx)
        if self.pydantic_agent:
            print("ExecuterAgent delegating to PydanticChatAgent")
            logger.info("ExecuterAgent delegating to PydanticChatAgent")
            return await self.pydantic_agent.run(ctx)
        if self.crewai_agent:
            print("ExecuterAgent delegating to CrewAIChatAgent")
            logger.info("ExecuterAgent delegating to CrewAIChatAgent")
            return await self.crewai_agent.run(ctx)
        return ChatAgentResponse(response="", tool_calls=[], citations=[])

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        if self.router_agent:
            print("ExecuterAgent streaming via CSORouterAgent")
            logger.info("ExecuterAgent streaming via CSORouterAgent")
            async for chunk in self.router_agent.run_stream(ctx):
                yield chunk
            return
        if self.pydantic_agent:
            print("ExecuterAgent streaming via PydanticChatAgent")
            logger.info("ExecuterAgent streaming via PydanticChatAgent")
            async for chunk in self.pydantic_agent.run_stream(ctx):
                yield chunk
            return
        if self.crewai_agent:
            print("ExecuterAgent streaming via CrewAIChatAgent")
            logger.info("ExecuterAgent streaming via CrewAIChatAgent")
            async for chunk in self.crewai_agent.run_stream(ctx):
                yield chunk
            return
