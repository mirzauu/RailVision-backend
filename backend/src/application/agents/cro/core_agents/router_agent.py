import logging
from typing import AsyncGenerator, Dict, TYPE_CHECKING

from pydantic import BaseModel, Field

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import ChatAgent, ChatAgentResponse, ChatContext, AgentConfig, TaskConfig
from src.infrastructure.agents.pydantic_multi_agent import PydanticMultiAgent

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from .general_agent import CROGeneralAgent
from .gabrial_agent import CROGabrialAgent
from .sales_performance_agent import SalesPerformanceAgent
from .pipeline_agent import PipelineAgent
from .market_expansion_agent import MarketExpansionAgent
from .pricing_strategy_agent import PricingStrategyAgent
from .partner_channel_agent import PartnerChannelAgent
from ..tool_agents.ppt_agent import CROPPTAgent
from ..tool_agents.pdf_agent import CROPDFAgent
from ..tool_agents.word_agent import CROWordAgent
from ..tool_agents.spreadsheet_agent import CROSpreadsheetAgent
from .brutall_agent import CROBrutallAgent
from ..sub_agents.micheal_agent import CROMichealAgent
from ..sub_agents.mary_agent import CROMaryAgent
from ..sub_agents.rapheal_agent import CRORaphealAgent
from ..sub_agents.emily_agent import CROEmilyAgent

logger = logging.getLogger(__name__)


class ClassificationResponse(BaseModel):
    agent_id: str = Field(description="agent_id of the best matching agent. Use 'multi_agent' if multiple agents are required.")
    confidence_score: float = Field(description="confidence score between 0 and 1")
    is_multi_agent: bool = Field(default=False, description="Set to True if the query requires coordination between multiple agents")


classification_prompt = (
    "You are part of the ai agentic system that routes the current query to the most appropriate CRO agent. "
    "Select the best agent by comparing the query’s requirements with each agent’s specialties.\n\n"
    "User Query: {query}\n"
    "Chat history: {history}\n"
    "--- end of Chat history ----\n\n"
    "Available agents and their specialties:\n"
    "{agent_descriptions}\n\n"
    "Analysis Instructions (do not include these in the final answer):\n"
    "1. Identify key topics, technical terms, and the user’s intent.\n"
    "2. Compare these elements to each agent’s specialty description.\n"
    "3. Favor specialized agents over general ones for close matches.\n"
    "4. MULTI-AGENT REQUIRED: If the query requires expertise from multiple different domains or combining insights from more than one agent, explicitly select 'multi_agent' as the agent_id and set is_multi_agent to True.\n\n"
    "Confidence Scoring Guidelines:\n"
    "- 0.9-1.0: Ideal match with core expertise.\n"
    "- 0.7-0.9: Strong match with known capabilities.\n"
    "- 0.5-0.7: Partial or related match.\n"
    "If multiple areas of expertise are needed, choose the 'multi_agent' option. If no agent is an ideal match, choose the best available option.\n"
)


CRO_SUPERVISOR_TASK_DESCRIPTION = """
You are Gabrial, the Head CRO of RailVision, operating in Multi-Agent Orchestration Mode.
You are not a passive coordinator — you are the commercial authority. You lead, synthesize, and take responsibility for the final output.

━━━━━━━━━━━━━━━━━━━━━━
STEP 1: UNDERSTAND THE QUERY
━━━━━━━━━━━━━━━━━━━━━━

Before calling any subagent or tool, deeply understand the user's intent:
- What is the user actually asking for?
- Is this a single-domain task or a multi-domain task?
- What is the minimum set of subagents and tools needed to answer this well?
- Are there dependencies between the subagents? (e.g., performance analysis must happen before a pipeline forecast)

━━━━━━━━━━━━━━━━━━━━━━
STEP 2: MANDATORY TODO TRACKING (FOR MULTI-TASK QUERIES)
━━━━━━━━━━━━━━━━━━━━━━

If the query involves multiple steps, multiple subagents, or the creation of deliverables (PDF, PPT, Word, Spreadsheet), you MUST use the todo system to plan and track progress BEFORE starting work.

**Create a todo list at the start:**
1. Use `create_todo` to create one todo item per subagent delegation or major step.
   - Example: "Consult performance agent for last quarter metrics"
   - Example: "Generate PPT presentation for board review"
2. Use `update_todo_status` to mark steps as in-progress or complete as you work through them.
3. Use `add_todo_note` to record key findings or decisions made during execution.
4. Use `get_todo_summary` at the end to confirm all items are resolved before delivering the final answer.

**Do NOT use the todo system for simple, single-step queries.**

━━━━━━━━━━━━━━━━━━━━━━
STEP 3: DELEGATE TO SUBAGENTS
━━━━━━━━━━━━━━━━━━━━━━

Use your delegate tools (consult_*_agent) to query specialized subagents. Each tool routes to a specific expert:
- `consult_gabrial_agent`: YOUR own deep revenue strategy reasoning — use this when you need to think critically before synthesizing.
- `consult_performance_agent`: Sales performance analysis, KPIs, win rates, quota attainment.
- `consult_pipeline_agent`: Pipeline management, forecasting, deal progression, bottleneck identification.
- `consult_expansion_agent`: Market expansion strategy, new territories, and revenue streams.
- `consult_pricing_agent`: Pricing models, discount strategies, and packaging optimization.
- `consult_channel_agent`: Partner and channel revenue management, alliances.
- `consult_brutall_agent`: The Ruthless Mentor — use this to stress-test your sales strategies and commercial assumptions.
- `consult_ppt_agent`: Generate PowerPoint presentations or slide decks.
- `consult_pdf_agent`: Generate polished PDF reports, briefs, or memos.
- `consult_word_agent`: Generate structured Word documents.
- `consult_spreadsheet_agent`: Generate Excel spreadsheets or tabular data exports.

**Delegation Rules:**
- Only delegate to agents whose expertise is directly needed.
- Do NOT over-delegate — unnecessary agent calls waste time and reduce quality.
- If a document is requested (PDF, PPT, etc.), always ensure the commercial content is finalized BEFORE calling the document agent.

━━━━━━━━━━━━━━━━━━━━━━
STEP 4: USE YOUR OWN TOOLS
━━━━━━━━━━━━━━━━━━━━━━

Beyond delegation, you have direct access to the following tools. Use them proactively:

- **`think`**: Use this for deep commercial reasoning, tradeoff analysis, and to deliberate before finalizing your answer. MANDATORY for complex or high-stakes queries.
- **`knowledge_base`**: Access internal RailVision information — always prefer this over assumptions.
- **`web_search_tool`**: Validate external market data, competitor intelligence, or pricing benchmarks.
- **`search_attachments`**: Extract specific facts and data points from documents the user has uploaded.
- **Todo Tools** (`create_todo`, `update_todo_status`, `add_todo_note`, `get_todo`, `list_todos`, `get_todo_summary`): Use for task planning and progress tracking.

━━━━━━━━━━━━━━━━━━━━━━
STEP 5: SYNTHESIZE & DELIVER
━━━━━━━━━━━━━━━━━━━━━━

After gathering all insights, synthesize them into ONE cohesive executive-ready answer:
- Lead with the most important finding or recommendation.
- Clearly distinguish: ✔ Verified Facts | ~ Estimated/Inferred | ⚠ Requires Validation.
- Flag any missing information or risks before presenting conclusions.
- If a document was generated (PDF, PPT, etc.), include the download link prominently.
- Do NOT just concatenate subagent outputs — synthesize, challenge, and refine them.

━━━━━━━━━━━━━━━━━━━━━━
CRITICAL REMINDERS
━━━━━━━━━━━━━━━━━━━━━━

- You are responsible for the quality of the final output, not just coordination.
- Never present weak reasoning as strong conclusions.
- The user is an executive. Treat every response as if it will be used in a real boardroom meeting.
"""


class CRORouterAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.agents: Dict[str, ChatAgent] = {
            "general": CROGeneralAgent(llm_provider, tools_provider),
            "gabrial": CROGabrialAgent(llm_provider, tools_provider),
            "performance": SalesPerformanceAgent(llm_provider, tools_provider),
            "pipeline": PipelineAgent(llm_provider, tools_provider),
            "expansion": MarketExpansionAgent(llm_provider, tools_provider),
            "pricing": PricingStrategyAgent(llm_provider, tools_provider),
            "channel": PartnerChannelAgent(llm_provider, tools_provider),
            "ppt": CROPPTAgent(llm_provider, tools_provider),
            "pdf": CROPDFAgent(llm_provider, tools_provider),
            "word": CROWordAgent(llm_provider, tools_provider),
            "spreadsheet": CROSpreadsheetAgent(llm_provider, tools_provider),
            "brutall": CROBrutallAgent(llm_provider, tools_provider),
            "micheal": CROMichealAgent(llm_provider, tools_provider),
            "mary": CROMaryAgent(llm_provider, tools_provider),
            "rapheal": CRORaphealAgent(llm_provider, tools_provider),
            "emily": CROEmilyAgent(llm_provider, tools_provider),
        }
        self.agent_descriptions_map: Dict[str, str] = {
            "general": "Handles greetings and simple open-ended revenue questions; acts as a friendly front-door assistant for the CRO system.",
            "gabrial": "High-trust, senior CRO strategy expert that challenges assumptions and provides executive-ready revenue insights with uncertainty signaling. Select for complex, high-stakes strategic inquiries.",
            "performance": "Analyzes complex sales data, KPIs, win rates, sales cycles, and quota attainment. Explains past and current revenue performance. Select when asked 'how did we perform last quarter' or 'show me our sales metrics.'",
            "pipeline": "Manages, forecasts, and optimizes the sales pipeline. Focuses on future revenue, deal progression, and identifying bottlenecks. Use when asked 'what is our Q4 forecast' or 'analyze the health of our current deals.'",
            "expansion": "Identifies and evaluates new market opportunities, territories, and revenue streams. Select when asked about 'growth into Europe', 'new industries to target', or 'market entry strategy.'",
            "pricing": "Optimizes product pricing models, discount strategies, and packaging to maximize revenue. Choose when asked 'should we raise our prices', 'analyze our competitive pricing', or 'can we change our discount structure.'",
            "channel": "Develops and manages channel partnerships, alliances, and indirect revenue streams. Select when asked about 'value-added resellers', 'partner performance', or 'channel conflict resolution.'",
            "ppt": "Creates world-class, board-ready PowerPoint presentations focusing on revenue performance, target quotas, and sales pipeline. Use for ANY request to create, generate, or build a PowerPoint or .pptx presentation.",
            "pdf": "Produces consultant-grade PDF documents focusing on revenue forecasting and go-to-market execution plans. Use for ANY request to create, generate, or build a PDF report.",
            "word": "Generates polished, deeply-researched Word (.docx) documents such as sales playbooks or pipeline strategy memos. Use for ANY request to create, generate, or build a Word document.",
            "spreadsheet": "Transforms structured CRM data and sales requests into professional Excel (.xlsx) workbooks. Use for ANY request to create, generate, or build an Excel spreadsheet or track deal pipelines in a table.",
            "brutall": "The Ruthless Mentor subagent. Use this to stress-test sales strategies and commercial assumptions. Select when the user needs critical, 'no-holds-barred' commercial feedback.",
            "micheal": "CSO Liaison subagent with deep knowledge of corporate strategy and enterprise value coordination.",
            "mary": "CCO Liaison subagent providing context on commercial reality, sales pipelines, and customer acquisition.",
            "rapheal": "CFO Liaison subagent focusing on financial metrics, budgeting, and capital allocation.",
            "emily": "CTO Liaison subagent providing context on technology roadmaps, engineering feasibility, and technical debt.",
        }

        self.agent_descriptions = "\n".join(
            [
                f"agent_id: {agent_id}\n description: {self.agent_descriptions_map[agent_id]}\n"
                for agent_id in self.agents
            ]
        )
        
        self.supervisor_config = AgentConfig(
            role="Gabrial – Head CRO & Multi-Agent Orchestrator",
            goal=(
                "Lead the coordination of specialized CRO subagents to deliver authoritative, executive-ready responses. "
                "You are not just a router — you are the commercial center of gravity. Every response must reflect Gabrial's "
                "high-trust, grounded, and decision-ready standard."
            ),
            backstory=(
                "You are Gabrial, the Chief Revenue Officer of RailVision, operating in multi-agent orchestration mode. "
                "You have deep knowledge of RailVision's revenue strategy, sales performance, and commercial landscape. "
                "In this mode, you are supported by a team of specialized subagents — each expert in their own domain. "
                "Your job is to coordinate these subagents intelligently, synthesize their outputs, challenge weak reasoning, "
                "and deliver a single, cohesive answer that is safe for executive decision-making. "
                "You think like a CEO advisor: you never allow overconfidence, you always distinguish facts from assumptions, "
                "and you expose risks before they become problems."
            ),
            tasks=[
                TaskConfig(
                    description=CRO_SUPERVISOR_TASK_DESCRIPTION,
                    expected_output=(
                        "A comprehensive, executive-ready response that integrates all subagent insights under Gabrial's "
                        "commercial authority. Clearly distinguishes verified facts, inferences, and assumptions. "
                        "All delegated tasks are tracked and any outstanding items are logged in the todo system."
                    ),
                )
            ],
        )

    async def _run_classification(self, ctx: ChatContext, agent_descriptions: str) -> ChatAgent:
        prompt = classification_prompt.format(
            query=ctx.query,
            history=", ".join(f"{m['role']}: {m['content']}" for m in ctx.history[-5:]),
            agent_descriptions=agent_descriptions,
        )
        messages = [
            {
                "role": "system",
                "content": "You are an expert agent classifier that routes queries to the most appropriate CRO agent.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            classification: ClassificationResponse = await self.llm_provider.call_llm_with_structured_output(
                messages=messages,
                output_schema=ClassificationResponse,
                config_type="inference",
            )
            logger.info("Classification result: %s", classification)
            selected_agent_id = classification.agent_id if classification and (classification.agent_id in self.agents or classification.agent_id == "multi_agent") else "general"
            
            # Check for multi-agent
            if classification.is_multi_agent or selected_agent_id == "multi_agent":
                logger.info("CRORouterAgent selected 'multi_agent' mode")
                return PydanticMultiAgent(
                    llm_provider=self.llm_provider,
                    config=self.supervisor_config,
                    tools=self.tools_provider.get_tools(
                        [
                            "think",
                            "create_todo",
                            "update_todo_status",
                            "add_todo_note",
                            "get_todo",
                            "list_todos",
                            "get_todo_summary",
                        ]
                    ),
                    existing_delegates=self.agents,
                    delegate_descriptions=self.agent_descriptions_map
                )

            logger.info(
                "CRORouterAgent selected '%s' with confidence %.2f",
                selected_agent_id,
                getattr(classification, "confidence_score", 0.0) or 0.0,
            )
            return self.agents[selected_agent_id]
        except Exception as e:
            logger.error("CRO Routing failed, falling back to general agent: %s", e)
            return self.agents["general"]

    def get_agent(self, agent_id: str) -> ChatAgent:
        return self.agents.get(agent_id) or self.agents["general"]

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        return await agent.run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        async for chunk in agent.run_stream(ctx):
            yield chunk
