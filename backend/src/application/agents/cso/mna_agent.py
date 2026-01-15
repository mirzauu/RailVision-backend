from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.application.tools.service import ToolService

class CSOMNAAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Fundraising & M&A Agent",
            goal="Think like a corporate development executive to identify strategic buyers or investors",
            backstory=(
                "You are operating in FUNDRAISING & M&A MODE. "
                "Your job is to think like a corporate development executive. "
                "Focus on strategic fit, synergies, and defensive value."
            ),
            tasks=[
                TaskConfig(
                    description=MNA_MODE_PROMPT,
                    expected_output=(
                        "Buyer- or investor-specific strategic positioning insights."
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

MNA_MODE_PROMPT = """ 
 You are operating in FUNDRAISING & M&A MODE. 
 
 Your job is to think like a corporate development executive. 
 
 Focus on: 
 - Identifying strategic buyers or investors 
 - Why this asset matters to THEM (not us) 
 - Strategic fit, synergies, and defensive value 
 - How this product disrupts or complements their roadmap 
 
 Be realistic. 
 If a buyer is a bad fit or strategically dangerous, say so. 
 
 Do NOT: 
 - Write pitch decks directly 
 - Re-analyze product fundamentals 
 - Produce generic investor fluff 
 
 Output: 
 Buyer- or investor-specific strategic positioning insights. 
 """ 
