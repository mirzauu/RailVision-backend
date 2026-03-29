import logging
from typing import AsyncGenerator, Dict, TYPE_CHECKING

from pydantic import BaseModel, Field

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import ChatAgent, ChatAgentResponse, ChatContext, AgentConfig, TaskConfig
from src.infrastructure.agents.pydantic_multi_agent import PydanticMultiAgent

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from .strategy_agent import CSOStrategyAgent
from .value_prop_agent import CSOValuePropAgent
from .gtm_agent import CSOGTMAgent
from .railroad_intel_agent import CSORailroadIntelAgent
from .mna_agent import CSOMNAAgent
from ..tool_agents.artifact_agent import CSOArtifactAgent
from ..tool_agents.ppt_agent import CSOPPTAgent
from ..tool_agents.pdf_agent import CSOPDFAgent
from ..tool_agents.word_agent import CSOWordAgent
from .general_agent import CSOGeneralAgent
from .brutall_agent import CSOBrutallAgent
from ..tool_agents.spreadsheet_agent import CSOSpreadsheetAgent
from .michael_agent import CSOMichaelAgent
from ..sub_agents.mary_agent import CSOMaryAgent
from ..sub_agents.rapheal_agent import CSORaphealAgent
from ..sub_agents.emily_agent import CSOEmilyAgent
from ..sub_agents.gabrial_agent import CSOGabrialAgent



logger = logging.getLogger(__name__)


class ClassificationResponse(BaseModel):
    agent_id: str = Field(description="agent_id of the best matching agent. Use 'multi_agent' if multiple agents are required.")
    confidence_score: float = Field(description="confidence score between 0 and 1")
    is_multi_agent: bool = Field(default=False, description="Set to True if the query requires coordination between multiple agents")


classification_prompt = (
    "You are part of the ai agentic system that routes the current query to the most appropriate CSO agent. "
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


SUPERVISOR_TASK_DESCRIPTION = """
You are Michael, the Head CSO of RailVision, operating in Multi-Agent Orchestration Mode.
You are not a passive coordinator — you are the strategic authority. You lead, synthesize, and take responsibility for the final output.

━━━━━━━━━━━━━━━━━━━━━━
STEP 1: UNDERSTAND THE QUERY
━━━━━━━━━━━━━━━━━━━━━━

Before calling any subagent or tool, deeply understand the user's intent:
- What is the user actually asking for?
- Is this a single-domain task or a multi-domain task?
- What is the minimum set of subagents and tools needed to answer this well?
- Are there dependencies between the subagents? (e.g., strategy analysis must happen before a PDF is generated)

━━━━━━━━━━━━━━━━━━━━━━
STEP 2: MANDATORY TODO TRACKING (FOR MULTI-TASK QUERIES)
━━━━━━━━━━━━━━━━━━━━━━

If the query involves multiple steps, multiple subagents, or the creation of deliverables (PDF, PPT, Word, Spreadsheet), you MUST use the todo system to plan and track progress BEFORE starting work.

**Create a todo list at the start:**
1. Use `create_todo` to create one todo item per subagent delegation or major step.
   - Example: "Consult strategy agent for competitive analysis"
   - Example: "Generate PDF report with findings"
2. Use `update_todo_status` to mark steps as in-progress or complete as you work through them.
3. Use `add_todo_note` to record key findings or decisions made during execution.
4. Use `get_todo_summary` at the end to confirm all items are resolved before delivering the final answer.

**Do NOT use the todo system for simple, single-step queries.**

━━━━━━━━━━━━━━━━━━━━━━
STEP 3: DELEGATE TO SUBAGENTS
━━━━━━━━━━━━━━━━━━━━━━

Use your delegate tools (consult_*_agent) to query specialized subagents. Each tool routes to a specific expert:
- `consult_michael_agent`: YOUR own deep strategic reasoning — use this when you need to think critically before synthesizing.
- `consult_strategy_agent`: Core strategy reduction, dominant constraints, asymmetric failure modes.
- `consult_mary_agent`: CCO perspective — commercial reality, revenue, customer acquisition, contracts.
- `consult_rapheal_agent`: CFO perspective — capital allocation, financial discipline, enterprise value.
- `consult_gtm_agent`: Go-to-market strategy, sales execution, market entry.
- `consult_railroad_intel_agent`: Rail industry domain expertise, technical standards, and operations.
- `consult_mna_agent`: M&A strategy, partnership evaluation, valuation.
- `consult_value_prop_agent`: Business case development, value propositions for RailVision.
- `consult_brutall_agent`: Ruthless stress-testing of ideas — use to pressure-test the plan before finalizing.
- `consult_ppt_agent`: Generate PowerPoint presentations or slide decks.
- `consult_pdf_agent`: Generate polished PDF reports, briefs, or memos.
- `consult_word_agent`: Generate structured Word documents.
- `consult_spreadsheet_agent`: Generate Excel spreadsheets or tabular data exports.

**Delegation Rules:**
- Only delegate to agents whose expertise is directly needed.
- Do NOT over-delegate — unnecessary agent calls waste time and reduce quality.
- If a document is requested (PDF, PPT, etc.), always ensure the strategic content is finalized BEFORE calling the document agent.

━━━━━━━━━━━━━━━━━━━━━━
STEP 4: USE YOUR OWN TOOLS
━━━━━━━━━━━━━━━━━━━━━━

Beyond delegation, you have direct access to the following tools. Use them proactively:

- **`think`**: Use this for deep strategic reasoning, tradeoff analysis, and to deliberate before finalizing your answer. MANDATORY for complex or high-stakes queries before submitting your final response.
- **`knowledge_base`**: Access internal RailVision information — always prefer this over assumptions when RailVision-specific facts are needed.
- **`web_search_tool`**: Validate external market data, competitor intelligence, or industry benchmarks.
- **`search_attachments`**: Extract specific facts and data points from documents the user has uploaded to the conversation.
- **Todo Tools** (`create_todo`, `update_todo_status`, `add_todo_note`, `get_todo`, `list_todos`, `get_todo_summary`): Use for task planning, progress tracking, and accountability on multi-step workflows.

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
- If a subagent's output feels incomplete or wrong, challenge it or consult another agent.
- Use `get_todo_summary` to verify all tracked tasks are complete before responding.
- The user is an executive. Treat every response as if it will be used in a real boardroom meeting.
"""


class CSORouterAgent(ChatAgent):

    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.agents: Dict[str, ChatAgent] = {
            "strategy": CSOStrategyAgent(llm_provider, tools_provider),
            "value_prop": CSOValuePropAgent(llm_provider, tools_provider),
            "gtm": CSOGTMAgent(llm_provider, tools_provider),
            "railroad_intel": CSORailroadIntelAgent(llm_provider, tools_provider),
            "mna": CSOMNAAgent(llm_provider, tools_provider),
            "artifact": CSOArtifactAgent(llm_provider, tools_provider),
            "ppt": CSOPPTAgent(llm_provider, tools_provider),
            "pdf": CSOPDFAgent(llm_provider, tools_provider),
            "word": CSOWordAgent(llm_provider, tools_provider),
            "spreadsheet": CSOSpreadsheetAgent(llm_provider, tools_provider),
            "general": CSOGeneralAgent(llm_provider, tools_provider),
            "michael": CSOMichaelAgent(llm_provider, tools_provider),
            "brutall": CSOBrutallAgent(llm_provider, tools_provider),
            "mary": CSOMaryAgent(llm_provider, tools_provider),
            "rapheal": CSORaphealAgent(llm_provider, tools_provider),
            "gabrial": CSOGabrialAgent(llm_provider, tools_provider),
            "emily": CSOEmilyAgent(llm_provider, tools_provider),
        }
        self.agent_descriptions_map: Dict[str, str] = {
            "ppt": "The primary agent for building PowerPoint (.pptx) slide decks; use this for ANY request involving slides, presentations, decks, or PowerPoint files.",
            "pdf": "Specialized in generating polished PDF reports and structured documents; use this for ANY request involving PDF reports, memos, briefs, or downloadable PDF documents.",
            "word": "Specialized in creating and updating structured Word documents and reports; use this for ANY request involving Word docs, reports, or memos.",
            "spreadsheet": "Specialized in generating Excel (.xlsx) spreadsheets with multiple sheets from structured data; use this for ANY request involving spreadsheets, Excel files, tabular data exports, or downloadable data files.",
            "general": "Handles greetings and simple open-ended questions; acts as a friendly front-door assistant for the CSO system.",
            "michael": "The ultimate strategic and technical authority with 10+ years of experience. Michael has exhaustive knowledge of RailVision technology and full tool access for high-stakes strategic challenges.",
            "mary": "The CCO liaison. Has deep knowledge of all Chief Commercial Officer (CCO) related topics including revenue growth, customer acquisition, contract execution, and go-to-market reality. Use this when the user specifically asks for Mary, CCO insights, or a commercial perspective on a strategic issue.",
            "rapheal": "The CFO liaison. Has deep knowledge of all Chief Financial Officer (CFO) related topics including capital allocation, financial planning, fiscal discipline, and enterprise value. Use this when the user specifically asks for Raphael, CFO insights, or a financial perspective on a strategic issue.",
            "gabrial": "The CRO liaison. Provides the Chief Revenue Officer perspective on pipeline management, revenue forecasting, and market share.",
            "emily": "The CTO liaison. Provides the Chief Technology Officer perspective on software architecture, engineering velocity, AI innovation, and system risks.",
        }

        self.agent_descriptions = "\n".join(
            [
                f"agent_id: {agent_id}\n description: {description}\n"
                for agent_id, description in self.agent_descriptions_map.items()
                if agent_id in self.agents
            ]
        )
        if not self.agent_descriptions:
            self.agent_descriptions = "No agents available for routing"
        
        self.supervisor_config = AgentConfig(
            role="Michael – Head CSO & Multi-Agent Orchestrator",
            goal=(
                "Lead the coordination of specialized CSO subagents to deliver authoritative, executive-ready responses. "
                "You are not just a router — you are the strategic center of gravity. Every response must reflect Michael's "
                "high-trust, grounded, and decision-ready standard."
            ),
            backstory=(
                "You are Michael, the Chief Strategy Officer of RailVision, operating in multi-agent orchestration mode. "
                "You have deep knowledge of RailVision's business, technology, and strategic landscape. "
                "In this mode, you are supported by a team of specialized subagents — each expert in their own domain. "
                "Your job is to coordinate these subagents intelligently, synthesize their outputs, challenge weak reasoning, "
                "and deliver a single, cohesive answer that is safe for executive decision-making. "
                "You think like a CEO advisor: you never allow overconfidence, you always distinguish facts from assumptions, "
                "and you expose risks before they become problems."
            ),
            tasks=[
                TaskConfig(
                    description=SUPERVISOR_TASK_DESCRIPTION,
                    expected_output=(
                        "A comprehensive, executive-ready response that integrates all subagent insights under Michael's "
                        "strategic authority. Clearly distinguishes verified facts, inferences, and assumptions. "
                        "All delegated tasks are tracked and any outstanding items are logged in the todo system."
                    ),
                )
            ],
        )
        
        logger.info(f"CSORouterAgent initialized with {len(self.agents)} agents")

    async def _run_classification(self, ctx: ChatContext, agent_descriptions: str) -> ChatAgent:
        prompt = classification_prompt.format(
            query=ctx.query,
            history=", ".join(f"{m['role']}: {m['content']}" for m in ctx.history[-5:]),
            agent_descriptions=agent_descriptions,
        )
        messages = [
            {
                "role": "system",
                "content": "You are an expert agent classifier that routes queries to the most appropriate CSO agent.",
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
            selected_agent_id = classification.agent_id if classification and (classification.agent_id in self.agents or classification.agent_id == "multi_agent") else "strategy"
            
            # Check for multi-agent
            if classification.is_multi_agent or selected_agent_id == "multi_agent":
                logger.info("CSORouterAgent selected 'multi_agent' mode")
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
                "CSORouterAgent selected '%s' with confidence %.2f",
                selected_agent_id,
                getattr(classification, "confidence_score", 0.0) or 0.0,
            )
            
        except Exception as e:
            logger.error("Classification error, falling back to strategy agent: %s", e)
            selected_agent_id = "strategy"
        return self.agents[selected_agent_id]

    def get_agent(self, agent_id: str) -> ChatAgent:
        """Directly retrieve an agent by ID without classification."""
        return self.agents.get(agent_id) or self.agents["strategy"]

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        return await agent.run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        async for chunk in agent.run_stream(ctx):
            yield chunk
