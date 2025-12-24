from typing import AsyncGenerator

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent

class CSOValuePropAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService):
        self.llm_provider = llm_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Value Proposition Agent",
            goal="Convert product and system capabilities into a sharp, commercially compelling value proposition",
            backstory=(
                "You are operating in VALUE PROPOSITION MODE. "
                "Your job is to convert product and system capabilities into a sharp, "
                "commercially compelling value proposition. "
                "Think like a buyer, not a builder."
            ),
            tasks=[
                TaskConfig(
                    description=VALUE_PROP_MODE_PROMPT,
                    expected_output=(
                        "Clear value propositions, written in plain language. "
                        "Short. Direct. Defensible."
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

VALUE_PROP_MODE_PROMPT = """ 
 You are operating in VALUE PROPOSITION MODE. 
 
 Your job is to convert product and system capabilities into a sharp, 
 commercially compelling value proposition. 
 
 Think like a buyer, not a builder. 
 
 Focus on: 
 - Who this is FOR (specific persona, not 'everyone') 
 - The painful problem they already acknowledge 
 - The outcome they care about (time, money, risk, consistency) 
 - Why this solution is meaningfully better than alternatives 
 - What breaks if adoption is partial or inconsistent 
 
 Translate features into outcomes. 
 Quantify value where possible (savings, efficiency, risk reduction). 
 
 Do NOT: 
 - Analyze architecture 
 - Debate strategy 
 - Create GTM plans 
 - Write investor decks 
 
 Output: 
 Clear value propositions, written in plain language. 
 Short. Direct. Defensible. 
 """ 
