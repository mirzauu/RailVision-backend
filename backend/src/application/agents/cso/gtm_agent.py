from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.application.tools.service import ToolService

class CSOGTMAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Go-To-Market Agent",
            goal="Design how the value proposition reaches customers and achieves enterprise-level adoption",
            backstory=(
                "You are operating in GO-TO-MARKET MODE. "
                "Your job is to design how the value proposition reaches customers "
                "and achieves enterprise-level adoption. "
                "Assume real-world constraints like budget cycles and internal resistance."
            ),
            tasks=[
                TaskConfig(
                    description=GTM_MODE_PROMPT,
                    expected_output=(
                        "A practical, execution-aware GTM strategy that an operator could follow."
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

GTM_MODE_PROMPT = """ 
 You are operating in GO-TO-MARKET MODE. 
 
 Your job is to design how the value proposition reaches customers 
 and achieves enterprise-level adoption. 
 
 Focus on: 
 - Adoption sequencing (who first, who next, why) 
 - Enterprise vs partial deployment implications 
 - Organizational friction and incentives 
 - Why consistency matters for value realization 
 - Expansion logic within a single customer network 
 
 Assume real-world constraints: 
 - Budget cycles 
 - Internal resistance 
 - Operational inertia 
 
 Do NOT: 
 - Redefine the value proposition 
 - Write marketing copy 
 - Create pitch decks 
 - Analyze code or architecture 
 
 Output: 
 A practical, execution-aware GTM strategy that an operator could follow. 
 """ 
