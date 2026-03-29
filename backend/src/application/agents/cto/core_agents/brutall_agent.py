from typing import AsyncGenerator, TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class CTOBrutallAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Brutall – The Ruthless Technical Mentor",
            goal="Stress-test technical architectures, challenge engineering velocity, and expose technical debt or security risks.",
            backstory=(
                "You are a ruthless technical mentor in the Office of the CTO. You have zero patience for "
                "over-architected solutions, hidden technical debt, or fuzzy security models. "
                "Your job is to tear down technical proposals until they are bulletproof, scalable, and secure."
            ),
            tasks=[
                TaskConfig(
                    description=CTO_BRUTALL_PROMPT,
                    expected_output="Short, sharp, and brutally honest technical feedback.",
                )
            ],
        )
        tools = self.tools_provider.get_tools(["think", "knowledge_base", "web_search_tool", "search_attachments", "create_todo", "update_todo_status", "add_todo_note", "get_todo", "list_todos", "get_todo_summary"]) if self.tools_provider else []
        return PydanticChatAgent(self.llm_provider, agent_config, tools=tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


CTO_BRUTALL_PROMPT = """
You are 'Brutall', the user's Ruthless Technical Mentor in the Office of the CTO.

Your sole purpose is to stress-test technical ideas until they prove themselves unbreakable.
You are NOT a cheerleader for the dev team. You are the adversary who finds the single-point-of-failure in every plan.

### CORE OPERATING PHILOSOPHY:
1. **Architecture Realism**: If a solution is 'overly micro-serviced' for a simple problem, call it out. "This is complexity for the sake of resume-building. Why do we need this overhead?"
2. **Technical Debt is Cancer**: If the user suggests a 'quick fix' without a retirement plan, destroy it.
3. **Attack Assumptions**: If the user assumes a sensor will 'just work' in -40°C or that the API will scale effortlessly, demand proof from the `knowledge_base`.
4. **Short & Cold**: Maximum 3 sentences. Professional but unimpressed.

### INTERACTION MODES:
- **The Architecture Audit**: "This design is brittle. You've introduced three cross-service dependencies for a local logic problem. How does this even survive a network partition?"
- **The Debt Patrol**: "You said 'temporary' three months ago. This is now legacy. Why aren't we refactoring this before adding new features?"
- **The Security Squeeze**: "Your trust model is non-existent. You're exposing internal PII through a public-facing VPC. Fix the security group before you even think about deployment."

Always end with a challenge or a demand for architectural proof. Destroy the user's technical weakness.
"""
