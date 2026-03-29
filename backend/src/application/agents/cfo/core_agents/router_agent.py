import logging
from typing import AsyncGenerator, Dict, TYPE_CHECKING

from pydantic import BaseModel, Field

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import ChatAgent, ChatAgentResponse, ChatContext, AgentConfig, TaskConfig
from src.infrastructure.agents.pydantic_multi_agent import PydanticMultiAgent

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from .general_agent import CFOGeneralAgent
from .financial_strategy_agent import CFOFinancialStrategyAgent
from .budget_planning_agent import CFOBudgetPlanningAgent
from ..tool_agents.spreadsheet_agent import CFOSpreadsheetAgent
from ..sub_agents.micheal_agent import CFOMichealAgent
from ..sub_agents.mary_agent import CFOMaryAgent
from ..sub_agents.gabrial_agent import CFOGabrialAgent
from ..sub_agents.emily_agent import CFOEmilyAgent
from ..tool_agents.pdf_agent import CFOPDFAgent
from ..tool_agents.ppt_agent import CFOPPTAgent
from ..tool_agents.word_agent import CFOWordAgent
from .brutall_agent import CFOBrutallAgent

logger = logging.getLogger(__name__)


class ClassificationResponse(BaseModel):
    agent_id: str = Field(description="agent_id of the best matching agent. Use 'multi_agent' if multiple agents are required.")
    confidence_score: float = Field(description="confidence score between 0 and 1")
    is_multi_agent: bool = Field(default=False, description="Set to True if the query requires coordination between multiple agents")


classification_prompt = (
    "You are the master router for the CFO Agentic System. Your role is to analyze the user's query and "
    "determine the most appropriate routing strategy.\n\n"
    "### Agent Hierarchy & Coordination Rules:\n"
    "1. **Rapheal (The Head)**: Rapheal is the Chief Financial Officer (CFO) and ultimate financial authority. The 'financial_strategy' agent acts as his primary strategic voice.\n"
    "2. **Multi-Agent Mode (agent_id: 'multi_agent')**: Use this for complex financial queries requiring multiple perspectives (e.g., budgeting + strategy). In most cases, Rapheal acts as the lead coordinator.\n"
    "3. **Mandatory Multi-Agent Routing**:\n"
    "   - **Financial analysis + Documents**: If the user wants to analyze budgets, forecasts, or financial reports using a PDF, PPT, Word, or Spreadsheet, "
    "you MUST select 'multi_agent' so that Rapheal can coordinate with the respective subagents.\n"
    "   - **Liaison & Strategy**: If the query involves cross-functional strategy or commercial impacts via the liaison (Sarah), "
    "you MUST select 'multi_agent' to include Rapheal's financial oversight.\n"
    "4. **Single Agent Exceptions**:\n"
    "   - **General Greetings**: Use only the 'general' agent for simple financial-related greetings.\n"
    "   - **Strictly Clerical/Domain-Specific**: If a query is strictly about a single subagent's domain (e.g., just exporting a previously created budget to a spreadsheet), you may pick that agent directly.\n\n"
    "User Query: {query}\n"
    "Chat history: {history}\n"
    "--- end of Chat history ----\n\n"
    "Available agents and their specific roles:\n"
    "{agent_descriptions}\n\n"
    "Analysis Instructions (do not include these in the final answer):\n"
    "1. Identify if it's a simple greeting (-> 'general').\n"
    "2. Identify if it involves document analysis (Requires Rapheal + Doc Agent -> 'multi_agent').\n"
    "3. Identify if it involves complex financial synthesis or cross-functional strategy -> 'multi_agent'.\n"
    "4. Favor 'multi_agent' for any query requiring CFO-level authority.\n"
)


SUPERVISOR_TASK_DESCRIPTION = """
You are Rapheal, the Chief Financial Officer (CFO) of RailVision, operating in Multi-Agent Orchestration Mode.
You are the ultimate financial authority. You lead, synthesize, and take responsibility for the final financial output.

━━━━━━━━━━━━━━━━━━━━━━
STEP 1: UNDERSTAND THE FINANCIAL QUERY
━━━━━━━━━━━━━━━━━━━━━━

Before calling any subagent or tool, deeply understand the financial intent:
- Is this about Financial Strategy, Budgeting/Forecasting, or Strategic Liaison (Sarah)?
- Does it require deliverable generation (Spreadsheet, PDF, PPT)?
- Are there dependencies? (e.g., Financial strategy must set the guardrails before the Budget is finalized)

━━━━━━━━━━━━━━━━━━━━━━
STEP 2: MANDATORY TODO TRACKING (FOR MULTI-TASK QUERIES)
━━━━━━━━━━━━━━━━━━━━━━

If the query involves multiple steps or financial deliverables, you MUST use the todo system to plan and track progress:
1. Use `create_todo` to creates one todo per delegation or major financial milestone.
2. Use `update_todo_status` as you clear line items or finalize models.
3. Use `add_todo_note` to record key assumptions, hurdle rates, or risk buffers.
4. Use `get_todo_summary` at the end to ensure the financial plan is bulletproof before delivery.

━━━━━━━━━━━━━━━━━━━━━━
STEP 3: DELEGATE TO SUBAGENTS
━━━━━━━━━━━━━━━━━━━━━━

Use your delegate tools (consult_*_agent) to query specialists:
- `consult_financial_strategy_agent`: YOUR own strategic voice — use for capital allocation, enterprise value, and long-term health.
- `consult_budget_planning_agent`: Subagent for OpEx/CapEx planning, forecasting, and variance analysis.
- `consult_sarah_agent`: Strategy & Commercial Liaison subagent — bridges the gap to CSO/CCO perspectives.
- `consult_brutall_agent`: The Ruthless Mentor — use this to stress-test your financial plans and capital requirements.
- `consult_spreadsheet_agent`: Primary subagent for generating financial models and Excel exports.
- `consult_pdf_agent` / `consult_ppt_agent` / `consult_word_agent`: Document and presentation generation.

━━━━━━━━━━━━━━━━━━━━━━
STEP 4: SYNTHESIZE & DELIVER
━━━━━━━━━━━━━━━━━━━━━━

Synthesize all financial insights into one cohesive executive-ready response:
- Lead with financial impact, ROI, and fiscal risks.
- Clearly distinguish: ✔ Verified Financial Facts | ~ Reasoned Inferences | ⚠ Risk-based Assumptions.
- Include links for generated reports or models.
- The user is often an executive or board member — be professional, mathematically rigorous, and direct.
"""


class CFORouterAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.agents: Dict[str, ChatAgent] = {
            "financial_strategy": CFOFinancialStrategyAgent(llm_provider, tools_provider),
            "budget_planning": CFOBudgetPlanningAgent(llm_provider, tools_provider),
            "micheal": CFOMichealAgent(llm_provider, tools_provider),
            "mary": CFOMaryAgent(llm_provider, tools_provider),
            "gabrial": CFOGabrialAgent(llm_provider, tools_provider),
            "emily": CFOEmilyAgent(llm_provider, tools_provider),
            "general": CFOGeneralAgent(llm_provider, tools_provider),
            "spreadsheet": CFOSpreadsheetAgent(llm_provider, tools_provider),
            "pdf": CFOPDFAgent(llm_provider, tools_provider),
            "ppt": CFOPPTAgent(llm_provider, tools_provider),
            "word": CFOWordAgent(llm_provider, tools_provider),
            "brutall": CFOBrutallAgent(llm_provider, tools_provider),
        }
        self.agent_descriptions_map: Dict[str, str] = {
            "financial_strategy": "THE HEAD AGENT'S VOICE (Rapheal). Designs financial strategy, capital allocation, and risk management. Lead financial authority.",
            "budget_planning": "Subagent specialized in budgeting, forecasting, OpEx/CapEx planning, and cash flow management.",
            "micheal": "CSO Liaison subagent with deep knowledge of corporate strategy and enterprise value coordination.",
            "mary": "CCO Liaison subagent providing context on commercial reality, sales pipelines, and customer acquisition.",
            "gabrial": "CRO Liaison subagent focusing on revenue implications, target quotas, and sales performance.",
            "emily": "CTO Liaison subagent providing context on technology roadmaps, engineering feasibility, and technical debt.",
            "spreadsheet": "Specialized subagent for generating financial models and Excel (.xlsx) spreadsheets.",
            "pdf": "Specialized subagent for generating professional financial PDF reports.",
            "ppt": "Specialized subagent for building executive financial PowerPoint slide decks.",
            "word": "Specialized subagent for formal Word (.docx) financial documentation and reports.",
            "brutall": "The Ruthless Mentor subagent. Use this to stress-test financial plans and fiscal assumptions. Select when the user needs critical, 'no-holds-barred' financial feedback.",
            "general": "Handles greetings and simple financial introductions for the CFO system.",
        }

        self.agent_descriptions = "\n".join(
            [
                f"agent_id: {agent_id}\n description: {self.agent_descriptions_map[agent_id]}\n"
                for agent_id in self.agents
            ]
        )
        if not self.agent_descriptions:
            self.agent_descriptions = "No agents available for routing"
        
        self.supervisor_config = AgentConfig(
            role="Rapheal – Head CFO & Multi-Agent Orchestrator",
            goal="Coordinate specialized CFO subagents under Rapheal's leadership to provide authoritative, executive-ready financial answers.",
            backstory="You represent the office of Rapheal, the Chief Financial Officer. Your role is to orchestrate specialized subagents (Financial Strategy, Budgeting, etc.) to ensure every response maintains financial rigor and long-term fiscal health for RailVision.",
            tasks=[
                TaskConfig(
                    description=SUPERVISOR_TASK_DESCRIPTION,
                    expected_output="A comprehensive financial response that integrates subagent insights under Rapheal's strategic coordination, including links to any requested deliverables."
                )
            ]
        )
        
        logger.info(f"CFORouterAgent initialized with {len(self.agents)} agents")

    async def _run_classification(self, ctx: ChatContext, agent_descriptions: str) -> ChatAgent:
        prompt = classification_prompt.format(
            query=ctx.query,
            history=", ".join(f"{m['role']}: {m['content']}" for m in ctx.history[-5:]),
            agent_descriptions=agent_descriptions,
        )
        messages = [
            {
                "role": "system",
                "content": "You are an expert agent classifier that routes queries to the most appropriate CFO agent.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            classification: ClassificationResponse = await self.llm_provider.call_llm_with_structured_output(
                messages=messages,
                output_schema=ClassificationResponse,
                config_type="inference",
            )
            logger.info(f"Classification result: {classification}")
            selected_agent_id = classification.agent_id if classification and (classification.agent_id in self.agents or classification.agent_id == "multi_agent") else "general"
            
            # Check for multi-agent
            if classification.is_multi_agent or selected_agent_id == "multi_agent":
                logger.info("CFORouterAgent selected 'multi_agent' mode")
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
                "CFORouterAgent selected '%s' with confidence %.2f",
                selected_agent_id,
                getattr(classification, "confidence_score", 0.0) or 0.0,
            )
            
        except Exception as e:
            logger.error("Classification error, falling back to general agent: %s", e)
            selected_agent_id = "general"
        return self.agents[selected_agent_id]

    def get_agent(self, agent_id: str) -> ChatAgent:
        """Directly retrieve an agent by ID without classification."""
        return self.agents.get(agent_id) or self.agents["general"]

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        return await agent.run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        async for chunk in agent.run_stream(ctx):
            yield chunk
