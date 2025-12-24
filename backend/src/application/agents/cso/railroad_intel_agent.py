from typing import AsyncGenerator

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent

class CSORailroadIntelAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService):
        self.llm_provider = llm_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Railroad Intelligence Agent",
            goal="Build and refine a mental model of a specific railroad as a living system",
            backstory=(
                "You are operating in RAILROAD INTELLIGENCE MODE. "
                "Your job is to build and refine a mental model of a specific railroad "
                "as a living system — not a generic customer."
            ),
            tasks=[
                TaskConfig(
                    description=RAILROAD_INTEL_MODE_PROMPT,
                    expected_output=(
                        "Clear, railroad-specific insights and recommendations. "
                        "Assume this knowledge will compound over time."
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

RAILROAD_INTEL_MODE_PROMPT = """ 
 You are operating in RAILROAD INTELLIGENCE MODE. 
 
 Your job is to build and refine a mental model of a specific railroad 
 as a living system — not a generic customer. 
 
 Focus on: 
 - Network structure and operational realities 
 - Decision-making dynamics 
 - Constraints (technical, political, cultural) 
 - What this railroad values most (cost, safety, consistency, speed) 
 - How adoption would realistically occur inside this organization 
 
 Continuously update understanding as new information appears. 
 
 Do NOT: 
 - Generalize across all railroads 
 - Produce marketing language 
 - Create investor materials 
 
 Output: 
 Clear, railroad-specific insights and recommendations. 
 Assume this knowledge will compound over time. 
 """ 
