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
from .spreadsheet_agent import CFOSpreadsheetAgent
from .sarah_agent import CFOSarahAgent
from .pdf_agent import CFOPDFAgent
from .ppt_agent import CFOPPTAgent
from .word_agent import CFOWordAgent

logger = logging.getLogger(__name__)


class ClassificationResponse(BaseModel):
    agent_id: str = Field(description="agent_id of the best matching agent. Use 'multi_agent' if multiple agents are required.")
    confidence_score: float = Field(description="confidence score between 0 and 1")
    is_multi_agent: bool = Field(default=False, description="Set to True if the query requires coordination between multiple agents")


classification_prompt = (
    "You are part of the ai agentic system that routes the current query to the most appropriate CFO agent. "
    "Select the best agent by comparing the query's requirements with each agent's specialties.\n\n"
    "User Query: {query}\n"
    "Chat history: {history}\n"
    "--- end of Chat history ----\n\n"
    "Available agents and their specialties:\n"
    "{agent_descriptions}\n\n"
    "Analysis Instructions (do not include these in the final answer):\n"
    "1. Identify key topics, technical terms, and the user's intent.\n"
    "2. Compare these elements to each agent's specialty description.\n"
    "3. Favor specialized agents over general ones for close matches.\n"
    "4. MULTI-AGENT REQUIRED: If the query requires expertise from multiple different domains or combining insights from more than one agent, explicitly select 'multi_agent' as the agent_id and set is_multi_agent to True.\n\n"
    "Confidence Scoring Guidelines:\n"
    "- 0.9-1.0: Ideal match with core expertise.\n"
    "- 0.7-0.9: Strong match with known capabilities.\n"
    "- 0.5-0.7: Partial or related match.\n"
    "If multiple areas of expertise are needed, choose the 'multi_agent' option. If no agent is an ideal match, choose the best available option.\n"
)


class CFORouterAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.agents: Dict[str, ChatAgent] = {
            "financial_strategy": CFOFinancialStrategyAgent(llm_provider, tools_provider),
            "budget_planning": CFOBudgetPlanningAgent(llm_provider, tools_provider),
            "sarah": CFOSarahAgent(llm_provider, tools_provider),
            "general": CFOGeneralAgent(llm_provider, tools_provider),
            "spreadsheet": CFOSpreadsheetAgent(llm_provider, tools_provider),
            "pdf": CFOPDFAgent(llm_provider, tools_provider),
            "ppt": CFOPPTAgent(llm_provider, tools_provider),
            "word": CFOWordAgent(llm_provider, tools_provider),
        }
        self.agent_descriptions_map: Dict[str, str] = {
            "financial_strategy": "Designs financial strategy, capital allocation, risk management, and long-term financial planning; focuses on enterprise value, capital structure, and financial health.",
            "budget_planning": "Leads budgeting, forecasting, and variance analysis; handles OpEx/CapEx planning, cash flow management, and financial performance tracking.",
            "sarah": "The strategy and commercial liaison within the CFO team. Bridges the gap between financial constraints and strategic/commercial opportunities.",
            "general": "Handles greetings and simple open-ended financial questions; acts as a friendly front-door assistant for the CFO system.",
            "spreadsheet": "Specialized in generating Excel (.xlsx) spreadsheets from financial data; use this for ANY request involving financial models, budgets, or tabular data exports.",
            "pdf": "Generates professional PDF financial reports and presentations; use this when the user specifically asks for a PDF version of financial documents.",
            "ppt": "Creates high-impact executive PowerPoint (.pptx) slide decks for financial presentations.",
            "word": "Produces polished Word (.docx) documents and formal financial reports; ideal for long-form textual financial documentation.",
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
            role="CFO Supervisor",
            goal="Coordinate specialized CFO agents to provide comprehensive financial answers.",
            backstory="You are the Chief of Staff to the CFO. You coordinate specialized agents (Financial Strategy, Budget Planning) to answer complex queries that require multiple financial perspectives.",
            tasks=[
                TaskConfig(
                    description="Analyze the user query and consult the appropriate specialized agents to provide a comprehensive answer.",
                    expected_output="A well-reasoned, comprehensive response that integrates insights from multiple specialized agents."
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
