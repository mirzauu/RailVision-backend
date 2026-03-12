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
from .artifact_agent import CSOArtifactAgent
from .ppt_agent import CSOPPTAgent

from .pdf_agent import CSOPDFAgent
from .word_agent import CSOWordAgent
from .general_agent import CSOGeneralAgent
from .brutall_agent import CSOBrutallAgent
from .spreadsheet_agent import CSOSpreadsheetAgent
from .mary_agent import CSOMaryAgent


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
            "brutall": CSOBrutallAgent(llm_provider, tools_provider),
            "mary": CSOMaryAgent(llm_provider, tools_provider),
        }
        self.agent_descriptions_map: Dict[str, str] = {
            "strategy": "Analyzes repository as a strategic asset; identifies business models, value creation, and strategic leverage vs constraints.",
            "value_prop": "Converts capabilities into sharp value propositions; focuses on buyer personas, painful problems, and outcomes.",
            "gtm": "Designs go-to-market strategies; focuses on adoption sequencing, enterprise deployment, and organizational friction.",
            "railroad_intel": "Builds mental models of specific railroads; focuses on network structure, decision dynamics, and operational constraints.",
            "mna": "Thinks like a corporate development executive; identifies strategic buyers, synergies, and defensive value.",
            "artifact": "Converts inputs into polished artifacts like memos, emails, briefings, and action plans.",
            "ppt": "The primary agent for building PowerPoint (.pptx) slide decks; use this for ANY request involving slides, presentations, decks, or PowerPoint files.",
            "pdf": "Specialized in generating polished PDF reports and structured documents; use this for ANY request involving PDF reports, memos, briefs, or downloadable PDF documents.",
            "word": "Specialized in creating and updating structured Word documents and reports; use this for ANY request involving Word docs, reports, or memos.",
            "spreadsheet": "Specialized in generating Excel (.xlsx) spreadsheets with multiple sheets from structured data; use this for ANY request involving spreadsheets, Excel files, tabular data exports, or downloadable data files.",
            "general": "Handles greetings and simple open-ended questions; acts as a friendly front-door assistant for the CSO system.",
            "brutall": "Ruthless mentor that challenges ideas, debates, and provides brutally honest feedback to test resilience. Use this when the user wants to be challenged or have their ideas torn apart.",
            "mary": "The CCO liaison. Has deep knowledge of all Chief Commercial Officer (CCO) related topics including revenue growth, customer acquisition, contract execution, and go-to-market reality. Use this when the user specifically asks for Mary, CCO insights, or a commercial perspective on a strategic issue.",
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
            role="CSO Supervisor",
            goal="Coordinate specialized CSO agents to provide comprehensive strategic answers.",
            backstory="You are the Chief of Staff to the CSO. You coordinate specialized agents (Strategy, GTM, M&A, etc.) to answer complex queries that require multiple perspectives.",
            tasks=[
                TaskConfig(
                    description="Analyze the user query and consult the appropriate specialized agents to provide a comprehensive answer.",
                    expected_output="A well-reasoned, comprehensive response that integrates insights from multiple specialized agents."
                )
            ]
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
