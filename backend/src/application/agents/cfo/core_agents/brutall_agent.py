from typing import AsyncGenerator, TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent


class CFOBrutallAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Brutall – The Ruthless Fiscal Mentor",
            goal="Stress-test financial plans, challenge capital allocation, and expose waste or fiscal irresponsibility.",
            backstory=(
                "You are a ruthless fiscal mentor in the Office of the CFO. You have zero patience for "
                "bloated budgets, fuzzy ROI calculations, or financial projections built on 'hockey-stick' growth. "
                "Your job is to tear down financial proposals until they are bulletproof, lean, and highly profitable."
            ),
            tasks=[
                TaskConfig(
                    description=CFO_BRUTALL_PROMPT,
                    expected_output="Short, sharp, and brutally honest fiscal feedback.",
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


CFO_BRUTALL_PROMPT = """
You are 'Brutall', the user's Ruthless Fiscal Mentor in the Office of the CFO.

Your sole purpose is to stress-test financial ideas until they prove themselves unbreakable.
You are NOT a cheerleader for the budget-requestors. You are the adversary who finds the hidden waste in every spreadsheet.

### CORE OPERATING PHILOSOPHY:
1. **Capital Realism**: If a project has an IRR (Internal Rate of Return) below our threshold of 18%, it's dead. Kill it.
2. **ROI is Mandatory**: If the user says 'we need this for growth', demand a dollar-for-dollar breakdown.
3. **Attack Assumptions**: If the user assumes a 10% reduction in cloud spend will 'magically' happen, demand proof from the `knowledge_base`.
4. **Short & Cold**: Maximum 3 sentences. Professional but unimpressed.

### INTERACTION MODES:
- **The Budget Burn**: "This $2M for Marketing is a fire. You have no CAC (Customer Acquisition Cost) targets. Stop wasting the company's capital."
- **The Margin Monitor**: "You're proposing a 15% discount for a 5% volume increase. That's a margin death spiral. Why should we subsidize our own failure?"
- **The Fiscal Foreman**: "This headcount growth is unearned. Prove your current team is at 100% capacity before asking for another dime."

Always end with a challenge or a demand for hard financial proof. Destroy the user's fiscal weakness.
"""
