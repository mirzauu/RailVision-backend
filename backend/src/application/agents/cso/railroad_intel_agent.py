from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.application.tools.service import ToolService

class CSORailroadIntelAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

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
        tools = self.tools_provider.get_tools(["think"]) if self.tools_provider else []
        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        enriched_query = await context_enrich(ctx.query, user_id=self.tools_provider.user_id) if self.tools_provider else ctx.query
        new_ctx = ctx.model_copy(update={"query": enriched_query})
        return await self._build_agent().run(new_ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        enriched_query = await context_enrich(ctx.query, user_id=self.tools_provider.user_id) if self.tools_provider else ctx.query
        new_ctx = ctx.model_copy(update={"query": enriched_query})
        async for chunk in self._build_agent().run_stream(new_ctx):
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
