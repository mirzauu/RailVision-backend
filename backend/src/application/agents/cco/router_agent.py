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


logger = logging.getLogger(__name__)


class ClassificationResponse(BaseModel):
    agent_id: str = Field(description="agent_id of the best matching agent. Use 'multi_agent' if multiple agents are required.")
    confidence_score: float = Field(description="confidence score between 0 and 1")
    is_multi_agent: bool = Field(default=False, description="Set to True if the query requires coordination between multiple agents")


classification_prompt = (
    "You are part of the ai agentic system that routes the current query to the most appropriate CCO agent. "
    "Select the best agent by comparing the query's requirements with each agent's specialties.\n\n"
    "User Query: {query}\n"
    "Chat history: {history}\n"
    "--- end of Chat history ----\n\n"
    "Available agents and their specialties:\n"
    "{agent_descriptions}\n\n"
    "Analysis Instructions (do not include these in the final answer):\n"
    "1. Identify key topics, technical terms, and the user's intent.\n"
    "2. Compare these elements to each agent's specialty description.\n"
    "3. Favor specialized agents over general ones for close matches.\n\n"
    "Confidence Scoring Guidelines:\n"
    "- 0.9-1.0: Ideal match with core expertise.\n"
    "- 0.7-0.9: Strong match with known capabilities.\n"
    "- 0.5-0.7: Partial or related match.\n"
    "If no agent is an ideal match, choose the best available option.\n"
)


class CCORouterAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.agents: Dict[str, ChatAgent] = {
            "sales_strategy": CCOSalesStrategyAgent(llm_provider, tools_provider),
            "contract": CCOContractAgent(llm_provider, tools_provider),
            "customer_success": CCOCustomerSuccessAgent(llm_provider, tools_provider),
            "general": CCOGeneralAgent(llm_provider, tools_provider),
        }
        self.agent_descriptions_map: Dict[str, str] = {
            "sales_strategy": "Designs commercial strategy, pricing architecture, packaging, and go-to-market plans for North American shortline railroads; focuses on value propositions, territory design, sales process optimization, and pipeline velocity.",
            "contract": "Leads end-to-end contract execution — from pilot-to-contract conversion through negotiation, deal structuring, signature, and renewal; handles multi-year agreements, risk management, and revenue protection.",
            "customer_success": "Builds and maintains senior-level customer relationships, drives account expansion and renewals, manages partner/channel development, and develops industry alliances (ASLRRA); focuses on retention, NRR, and customer advocacy.",
            "general": "Handles greetings and simple open-ended commercial questions; acts as a friendly front-door assistant for the CCO system, triaging queries to the right specialist.",
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
            role="CCO Supervisor",
            goal="Coordinate specialized CCO agents to provide comprehensive commercial answers.",
            backstory="You are the Chief of Staff to the CCO. You coordinate specialized agents (Sales Strategy, Contract, Customer Success) to answer complex queries that require multiple commercial perspectives.",
            tasks=[
                TaskConfig(
                    description="Analyze the user query and consult the appropriate specialized agents to provide a comprehensive answer.",
                    expected_output="A well-reasoned, comprehensive response that integrates insights from multiple specialized agents."
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
