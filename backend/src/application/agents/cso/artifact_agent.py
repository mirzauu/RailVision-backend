from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.application.tools.service import ToolService

class CSOArtifactAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

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
