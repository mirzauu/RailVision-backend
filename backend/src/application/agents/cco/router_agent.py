import logging
from typing import AsyncGenerator, Dict, TYPE_CHECKING

from pydantic import BaseModel, Field

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import ChatAgent, ChatAgentResponse, ChatContext, AgentConfig, TaskConfig
from src.infrastructure.agents.pydantic_multi_agent import PydanticMultiAgent

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from .sales_strategy_agent import CCOSalesStrategyAgent
from .customer_success_agent import CCOCustomerSuccessAgent
from .contract_agent import CCOContractAgent
from .general_agent import CCOGeneralAgent
from .ppt_agent import CCOPPTAgent
from .pdf_agent import CCOPDFAgent
from .word_agent import CCOWordAgent
from .spreadsheet_agent import CCOSpreadsheetAgent
from .brutall_agent import CCOBrutallAgent
from .micheal_agent import CCOMichealAgent

logger = logging.getLogger(__name__)


class ClassificationResponse(BaseModel):
    agent_id: str = Field(description="agent_id of the best matching agent. Use 'multi_agent' if multiple agents are required.")
    confidence_score: float = Field(description="confidence score between 0 and 1")
    is_multi_agent: bool = Field(default=False, description="Set to True if the query requires coordination between multiple agents")


classification_prompt = (
    "You are the master router for the CCO Agentic System. Your role is to analyze the user's query and "
    "determine the most appropriate routing strategy.\n\n"
    "### Agent Hierarchy & Coordination Rules:\n"
    "1. **Mary (The Head)**: Mary is the Chief Commercial Officer (CCO) and ultimate commercial authority. The 'sales_strategy' agent acts as her primary strategic voice.\n"
    "2. **Multi-Agent Mode (agent_id: 'multi_agent')**: Use this for complex commercial queries requiring multiple specialties (e.g., pricing + contracts). In most cases, Mary acts as the lead coordinator.\n"
    "3. **Mandatory Multi-Agent Routing**:\n"
    "   - **Commercial Analysis + Documents**: If the user wants to analyze sales, contracts, or customer data using a PDF, PPT, Word, or Spreadsheet, "
    "you MUST select 'multi_agent' so that Mary can coordinate with the respective document subagent.\n"
    "   - **Strategic Oversight**: If the user asks for Micheal (CSO Liaison) or complex strategy work, "
    "you MUST select 'multi_agent' to include Mary's commercial perspective along with strategic oversight.\n"
    "4. **Single Agent Exceptions**:\n"
    "   - **General Greetings**: Use only the 'general' agent for simple commercial-related greetings.\n"
    "   - **Strictly Domain-Specific**: If a query is strictly about a single subagent's domain (e.g., just updating a contract draft) without needing new strategy, you may pick that agent directly.\n\n"
    "User Query: {query}\n"
    "Chat history: {history}\n"
    "--- end of Chat history ----\n\n"
    "Available agents and their specific roles:\n"
    "{agent_descriptions}\n\n"
    "Analysis Instructions (do not include these in the final answer):\n"
    "1. Identify if it's a simple greeting (-> 'general').\n"
    "2. Identify if it involves document analysis (Requires Mary + Doc Agent -> 'multi_agent').\n"
    "3. Identify if it involves cross-functional coordination (Sales + Contract, or Commercial + CSO) -> 'multi_agent'.\n"
    "4. Favor 'multi_agent' for any query requiring CCO-level synthesis.\n"
)


SUPERVISOR_TASK_DESCRIPTION = """
You are Mary, the Chief Commercial Officer (CCO) of RailVision, operating in Multi-Agent Orchestration Mode.
You are the senior commercial authority. You lead, synthesize, and take responsibility for the final commercial output.

━━━━━━━━━━━━━━━━━━━━━━
STEP 1: UNDERSTAND THE COMMERCIAL QUERY
━━━━━━━━━━━━━━━━━━━━━━

Before calling any subagent or tool, deeply understand the commercial intent:
- Is this about Sales Strategy, Contract Negotiation, or Customer Success?
- Does it require document generation (PDF, PPT, Word, Spreadsheet)?
- Are there dependencies? (e.g., Pricing ROI must be verified before the Contract is drafted)

━━━━━━━━━━━━━━━━━━━━━━
STEP 2: MANDATORY TODO TRACKING (FOR MULTI-TASK QUERIES)
━━━━━━━━━━━━━━━━━━━━━━

If the query involves multiple steps or deliverables, you MUST use the todo system to plan and track progress:
1. Use `create_todo` to create one todo per delegation or major step.
2. Use `update_todo_status` as you complete commercial milestones.
3. Use `add_todo_note` to record negotiation points, pricing decisions, or customer risks.
4. Use `get_todo_summary` at the end to ensure the deal is fully structured before delivery.

━━━━━━━━━━━━━━━━━━━━━━
STEP 3: DELEGATE TO SUBAGENTS
━━━━━━━━━━━━━━━━━━━━━━

Use your delegate tools (consult_*_agent) to query specialists:
- `consult_sales_strategy_agent`: YOUR own strategic voice — use for pricing, GTM, and value prop design.
- `consult_contract_agent`: Subagent for end-to-end contract execution and negotiation.
- `consult_customer_success_agent`: Subagent for senior relationship management and account expansion.
- `consult_micheal_agent`: CSO Liaison subagent — use for strategic alignment and railroad domain expertise.
- `consult_brutall_agent`: Ruthless mentor — use to pressure-test the commercial viability of a deal.
- `consult_pdf_agent` / `consult_ppt_agent` / `consult_word_agent` / `consult_spreadsheet_agent`: Document generation.

━━━━━━━━━━━━━━━━━━━━━━
STEP 4: SYNTHESIZE & DELIVER
━━━━━━━━━━━━━━━━━━━━━━

Synthesize all commercial insights into one cohesive executive-ready response:
- Lead with revenue impact and commercial risk.
- Clearly distinguish: ✔ Verified Commercial Facts | ~ Reasoned Inferences | ⚠ Commercial Assumptions.
- Include links for generated contracts or presentations.
- The user is often an executive or board member — be professional, direct, and ROI-focused.
"""


class CCORouterAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.agents: Dict[str, ChatAgent] = {
            "sales_strategy": CCOSalesStrategyAgent(llm_provider, tools_provider),
            "contract": CCOContractAgent(llm_provider, tools_provider),
            "customer_success": CCOCustomerSuccessAgent(llm_provider, tools_provider),
            "ppt": CCOPPTAgent(llm_provider, tools_provider),
            "pdf": CCOPDFAgent(llm_provider, tools_provider),
            "word": CCOWordAgent(llm_provider, tools_provider),
            "spreadsheet": CCOSpreadsheetAgent(llm_provider, tools_provider),
            "general": CCOGeneralAgent(llm_provider, tools_provider),
            "brutall": CCOBrutallAgent(llm_provider, tools_provider),
            "micheal": CCOMichealAgent(llm_provider, tools_provider),
        }
        self.agent_descriptions_map: Dict[str, str] = {
            "sales_strategy": "THE HEAD AGENT'S VOICE (Mary). Designs commercial strategy, pricing architecture, packaging, and GTM plans. Lead commercial authority.",
            "contract": "Subagent specialized in contract execution: negotiation, deal structuring, and revenue protection.",
            "customer_success": "Subagent specialized in senior-level relationships, account expansion, and customer advocacy.",
            "micheal": "CSO Liaison subagent with deep knowledge of corporate strategy and railroad intelligence.",
            "ppt": "Specialized subagent for building commercial PowerPoint slide decks.",
            "pdf": "Specialized subagent for generating polished commercial PDF reports and briefs.",
            "word": "Specialized subagent for creating and updating commercial Word documents.",
            "spreadsheet": "Specialized subagent for generating Excel (.xlsx) spreadsheets from commercial data.",
            "general": "Handles greetings and simple commercial introductions for the CCO system.",
            "brutall": "The Ruthless Mentor subagent. Pressure-tests sales strategies and commercial assumptions.",
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
            role="Mary – Head CCO & Multi-Agent Orchestrator",
            goal="Coordinate specialized CCO subagents under Mary's leadership to provide authoritative, executive-ready commercial answers.",
            backstory="You represent the office of Mary, the Chief Commercial Officer. Your role is to orchestrate specialized subagents (Sales Strategy, Contract, Customer Success) to ensure every response aligns with commercial reality and drives revenue for RailVision.",
            tasks=[
                TaskConfig(
                    description=SUPERVISOR_TASK_DESCRIPTION,
                    expected_output="A comprehensive commercial response that integrates subagent insights under Mary's strategic coordination, including links to any requested deliverables."
                )
            ]
        )
        
        logger.info(f"CCORouterAgent initialized with {len(self.agents)} agents")

    async def _run_classification(self, ctx: ChatContext, agent_descriptions: str) -> ChatAgent:
        prompt = classification_prompt.format(
            query=ctx.query,
            history=", ".join(f"{m['role']}: {m['content']}" for m in ctx.history[-5:]),
            agent_descriptions=agent_descriptions,
        )
        messages = [
            {
                "role": "system",
                "content": "You are an expert agent classifier that routes queries to the most appropriate CCO agent.",
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
            selected_agent_id = classification.agent_id if classification and (classification.agent_id in self.agents or classification.agent_id == "multi_agent") else "sales_strategy"
            
            # Check for multi-agent
            if classification.is_multi_agent or selected_agent_id == "multi_agent":
                logger.info("CCORouterAgent selected 'multi_agent' mode")
                return PydanticMultiAgent(
                    llm_provider=self.llm_provider,
                    config=self.supervisor_config,
                    existing_delegates=self.agents,
                    delegate_descriptions=self.agent_descriptions_map
                )

            logger.info(
                "CCORouterAgent selected '%s' with confidence %.2f",
                selected_agent_id,
                getattr(classification, "confidence_score", 0.0) or 0.0,
            )
            
        except Exception as e:
            logger.error("Classification error, falling back to sales_strategy agent: %s", e)
            selected_agent_id = "sales_strategy"
        return self.agents[selected_agent_id]

    def get_agent(self, agent_id: str) -> ChatAgent:
        """Directly retrieve an agent by ID without classification."""
        return self.agents.get(agent_id) or self.agents["sales_strategy"]

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        return await agent.run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        async for chunk in agent.run_stream(ctx):
            yield chunk
