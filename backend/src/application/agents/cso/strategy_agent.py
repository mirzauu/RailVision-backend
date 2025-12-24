from typing import AsyncGenerator

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent

class CSOStrategyAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService):
        self.llm_provider = llm_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Strategy Agent",
            goal="Reason, diagnose, and challenge business strategy and assumptions",
            backstory=(
                "You are a Chief Strategy Officer-level thinker operating in STRATEGY MODE. "
                "Your job is to reason, diagnose, and challenge — not to sell, pitch, or format. "
                "You analyze the provided materials as a strategic asset rather than an implementation."
            ),
            tasks=[
                TaskConfig(
                    description=STRATEGY_MODE_PROMPT,
                    expected_output=(
                        "A concise but deep strategic analysis written for an executive audience. "
                        "Surface risks, trade-offs, and second-order effects."
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

STRATEGY_MODE_PROMPT = """ 
 You are operating in STRATEGY MODE. 
 
 You are a Chief Strategy Officer-level thinker. 
 Your job is to reason, diagnose, and challenge — not to sell, pitch, or format. 
 
 Analyze the provided materials (documents, architecture, product description, or repository) 
 as a strategic asset rather than an implementation. 
 
 Focus on: 
 - The implied business model and target customer 
 - The real value being created (not claimed) 
 - Strategic leverage vs long-term constraints 
 - Scalability, defensibility, and execution risk 
 - Hidden assumptions that could break under growth 
 - Signals of misalignment between technical choices and business goals 
 
 Do NOT: 
 - Write marketing copy 
 - Suggest go-to-market tactics 
 - Create decks, pitches, or artifacts 
 - Optimize language for persuasion 
 
 Output: 
 A concise but deep strategic analysis written for an executive audience. 
 Surface risks, trade-offs, and second-order effects. 
 Be direct. If something is weak, say so. 
 """ 
