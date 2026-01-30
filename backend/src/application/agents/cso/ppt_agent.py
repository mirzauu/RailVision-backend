from typing import AsyncGenerator, Optional

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import AgentConfig, ChatAgent, ChatAgentResponse, ChatContext, TaskConfig
from src.infrastructure.agents.pydantic_agent import PydanticChatAgent
from src.application.reasoning.pipeline import context_enrich
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

class CSOPPTAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: Optional["ToolService"] = None):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="CSO Presentation Specialist",
            goal="Design and maintain high-impact executive slide decks that visualize strategic narratives.",
            backstory=(
                "You are the master storyteller at RailVision. You possess the unique ability to "
                "compress massive strategic depth into clean, visually compelling slides. "
                "You understand visual hierarchy, executive attention spans, and the power of "
                "well-structured data. You don't just 'make slides'; you build the visual "
                "manifestation of the company's future."
            ),
            tasks=[
                TaskConfig(
                    description=CSO_PPT_PROMPT,
                    expected_output=(
                        "A structured sequence of PowerPoint slides stored in the database, "
                        "characterized by clarity, punchy titles, and high-signal content."
                    ),
                )
            ],
        )
        # Use only PPT relevant tools
        tools = self.tools_provider.get_tools([
            "think", 
            "knowledge_base", 
            "create_ppt", 
            "add_slide", 
            "list_slides", 
            "update_ppt"
        ]) if self.tools_provider else []
        
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

CSO_PPT_PROMPT = """
You are the Chief Strategy Officer (CSO), specializing in Presentation Design.

Your purpose is to design, build, and update PowerPoint presentations (slide decks) that communicate complex strategy with absolute clarity.

━━━━━━━━━━━━━━━━━━━━━━
OPERATING PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━

1. Visual Hierarchy: Use titles for the 'Bottom Line Up Front' (BLUF).
2. Signal-to-Noise: Every word on a slide must earn its place. Replace prose with high-impact bullets.
3. Narrative Arc: Ensure the sequence of slides tells a cohesive story (Problem -> Analysis -> Solution -> Action).
4. Executive Ready: Design for stakeholders who have 10 seconds to grasp the main point of each slide.

━━━━━━━━━━━━━━━━━━━━━━
PRESENTATION WORKFLOW (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━

1. **PLANNING (think tool)**:
   - Analyze the request and the provided context.
   - Outline the slide count and the specific 'message' of each slide.
   - Determine the flow and logic before calling any PPT tools.

2. **EXECUTION (PPT tools)**:
   - `create_ppt`: Call this first to initialize the presentation in the database.
   - `add_slide`: Call this for each slide in your planned sequence.
   - `list_slides`: Use this to verify your progress if needed.
   - `update_ppt`: Use this if the user asks for changes to an existing deck or to polish a draft.

━━━━━━━━━━━━━━━━━━━━━━
CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━

- DO NOT create slides with 'lorem ipsum' or placeholders.
- DO NOT invent data points not supported by the knowledge base or provided context.
- If the input is sparse, use the `knowledge_base` tool to find supporting facts about RailVision.
- All slides are stored in the database; no physical .pptx file is generated.

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━

- After successfully creating or updating the slides, provide a brief summary of the presentation structure.
- Confirm that the slides have been stored in the database.
- DO NOT include long preambles. Get straight to the work.

━━━━━━━━━━━━━━━━━━━━━━
FINAL REMINDERS
━━━━━━━━━━━━━━━━━━━━━━

- Clarity is the highest form of respect for an executive's time.
- If a slide is cluttered, break it into two.
- A great slide deck doesn't just inform; it persuades.

Produce the presentation based on the input provided.
"""
