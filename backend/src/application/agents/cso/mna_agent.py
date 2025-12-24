from typing import AsyncGenerator

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent

class CSOMNAAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService):
        self.llm_provider = llm_provider

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
        return PydanticChatAgent(self.llm_provider, agent_config, tools=[])

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
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
