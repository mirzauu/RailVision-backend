from typing import AsyncGenerator

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent

class CSOArtifactAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService):
        self.llm_provider = llm_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Artifact Agent",
            goal="Convert structured inputs into polished, executive-ready communication artifacts",
            backstory=(
                "You are operating in ARTIFACT MODE. "
                "Your job is to convert structured inputs into polished, "
                "executive-ready communication artifacts. "
                "You do NOT generate new strategy. You only organize, clarify, and sharpen language."
            ),
            tasks=[
                TaskConfig(
                    description=ARTIFACT_MODE_PROMPT,
                    expected_output=(
                        "Clean, concise, ready-to-use material."
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

ARTIFACT_MODE_PROMPT = """ 
 You are operating in ARTIFACT MODE. 
 
 Your job is to convert structured inputs into polished, 
 executive-ready communication artifacts. 
 
 You do NOT generate new strategy. 
 You do NOT challenge assumptions. 
 You do NOT explore alternatives. 
 
 You only: 
 - Organize 
 - Clarify 
 - Sharpen language 
 - Present information cleanly 
 
 Write in first person where appropriate. 
 Assume the speaker is a credible industry veteran. 
 
 Output: 
 Clean, concise, ready-to-use material. 
 """ 
