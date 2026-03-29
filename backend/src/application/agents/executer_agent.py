from typing import AsyncGenerator, Optional
import logging

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import ChatAgent, ChatAgentResponse, ChatContext, AgentConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.infrastructure.agents.crewai_agent import CrewAIChatAgent
from src.application.agents.cso.core_agents.router_agent import CSORouterAgent
from src.application.agents.cco.core_agents.router_agent import CCORouterAgent
from src.application.agents.cfo.core_agents.router_agent import CFORouterAgent
from src.application.agents.cto.core_agents.router_agent import CTORouterAgent
from src.application.agents.cro.core_agents.router_agent import CRORouterAgent
from src.application.tools.service import ToolService

logger = logging.getLogger(__name__)


class ExecuterAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, config: AgentConfig, framework: str = "pydantic", tools_provider: Optional[ToolService] = None):
        self.framework = framework.lower()
        self.pydantic_agent: Optional[PydanticChatAgent] = None
        self.crewai_agent: Optional[CrewAIChatAgent] = None
        self.router_agent: Optional[CSORouterAgent] = None
        self.cco_router_agent: Optional[CCORouterAgent] = None
        self.cfo_router_agent: Optional[CFORouterAgent] = None
        self.cto_router_agent: Optional[CTORouterAgent] = None
        self.cro_router_agent: Optional[CRORouterAgent] = None
        if self.framework == "pydantic":
            self.pydantic_agent = PydanticChatAgent(llm_provider, config)
        elif self.framework == "crewai":
            self.crewai_agent = CrewAIChatAgent(config)
        elif self.framework in {"router", "cso_router", "cso"}:
            if not tools_provider:
                raise ValueError("tools_provider is required for router framework")
            self.router_agent = CSORouterAgent(llm_provider, tools_provider)
        elif self.framework in {"cco_router", "cco"}:
            if not tools_provider:
                raise ValueError("tools_provider is required for CCO router framework")
            self.cco_router_agent = CCORouterAgent(llm_provider, tools_provider)
        elif self.framework in {"cfo_router", "cfo"}:
            if not tools_provider:
                raise ValueError("tools_provider is required for CFO router framework")
            self.cfo_router_agent = CFORouterAgent(llm_provider, tools_provider)
        elif self.framework in {"cto_router", "cto"}:
            if not tools_provider:
                raise ValueError("tools_provider is required for CTO router framework")
            self.cto_router_agent = CTORouterAgent(llm_provider, tools_provider)
        elif self.framework in {"cro_router", "cro"}:
            if not tools_provider:
                raise ValueError("tools_provider is required for CRO router framework")
            self.cro_router_agent = CRORouterAgent(llm_provider, tools_provider)
        else:
            self.pydantic_agent = PydanticChatAgent(llm_provider, config)
        
        chosen = "none"
        if self.cco_router_agent: chosen = "cco_router"
        elif self.cfo_router_agent: chosen = "cfo_router"
        elif self.cto_router_agent: chosen = "cto_router"
        elif self.cro_router_agent: chosen = "cro_router"
        elif self.router_agent: chosen = "router"
        elif self.pydantic_agent: chosen = "pydantic"
        elif self.crewai_agent: chosen = "crewai"
        
        logger.info("ExecuterAgent initialized using framework '%s' -> '%s'", self.framework, chosen)
        print(f"ExecuterAgent initialized using framework '{self.framework}' -> '{chosen}'")

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        if self.cco_router_agent:
            print("ExecuterAgent delegating to CCORouterAgent")
            logger.info("ExecuterAgent delegating to CCORouterAgent")
            return await self.cco_router_agent.run(ctx)
        if self.cfo_router_agent:
            print("ExecuterAgent delegating to CFORouterAgent")
            logger.info("ExecuterAgent delegating to CFORouterAgent")
            return await self.cfo_router_agent.run(ctx)
        if self.cto_router_agent:
            print("ExecuterAgent delegating to CTORouterAgent")
            logger.info("ExecuterAgent delegating to CTORouterAgent")
            return await self.cto_router_agent.run(ctx)
        if self.cro_router_agent:
            print("ExecuterAgent delegating to CRORouterAgent")
            logger.info("ExecuterAgent delegating to CRORouterAgent")
            return await self.cro_router_agent.run(ctx)
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
        if self.cco_router_agent:
            print("ExecuterAgent streaming via CCORouterAgent")
            logger.info("ExecuterAgent streaming via CCORouterAgent")
            async for chunk in self.cco_router_agent.run_stream(ctx):
                yield chunk
            return
        if self.cfo_router_agent:
            print("ExecuterAgent streaming via CFORouterAgent")
            logger.info("ExecuterAgent streaming via CFORouterAgent")
            async for chunk in self.cfo_router_agent.run_stream(ctx):
                yield chunk
            return
        if self.cto_router_agent:
            print("ExecuterAgent streaming via CTORouterAgent")
            logger.info("ExecuterAgent streaming via CTORouterAgent")
            async for chunk in self.cto_router_agent.run_stream(ctx):
                yield chunk
            return
        if self.cro_router_agent:
            print("ExecuterAgent streaming via CRORouterAgent")
            logger.info("ExecuterAgent streaming via CRORouterAgent")
            async for chunk in self.cro_router_agent.run_stream(ctx):
                yield chunk
            return
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
