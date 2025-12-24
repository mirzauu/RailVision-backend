import logging
from typing import AsyncGenerator, Dict

from pydantic import BaseModel, Field

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import ChatAgent, ChatAgentResponse, ChatContext
from .strategy_agent import CSOStrategyAgent
from .value_prop_agent import CSOValuePropAgent
from .gtm_agent import CSOGTMAgent
from .railroad_intel_agent import CSORailroadIntelAgent
from .mna_agent import CSOMNAAgent
from .artifact_agent import CSOArtifactAgent


logger = logging.getLogger(__name__)


class ClassificationResponse(BaseModel):
    agent_id: str = Field(description="agent_id of the best matching agent")
    confidence_score: float = Field(description="confidence score between 0 and 1")


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
    "3. Favor specialized agents over general ones for close matches.\n\n"
    "Confidence Scoring Guidelines:\n"
    "- 0.9-1.0: Ideal match with core expertise.\n"
    "- 0.7-0.9: Strong match with known capabilities.\n"
    "- 0.5-0.7: Partial or related match.\n"
    "If no agent is an ideal match, choose the best available option.\n"
)


class CSORouterAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService):
        self.llm_provider = llm_provider
        self.agents: Dict[str, ChatAgent] = {
            "strategy": CSOStrategyAgent(llm_provider),
            "value_prop": CSOValuePropAgent(llm_provider),
            "gtm": CSOGTMAgent(llm_provider),
            "railroad_intel": CSORailroadIntelAgent(llm_provider),
            "mna": CSOMNAAgent(llm_provider),
            "artifact": CSOArtifactAgent(llm_provider),
        }
        self.agent_descriptions_map: Dict[str, str] = {
            "strategy": "Analyzes repository as a strategic asset; identifies business models, value creation, and strategic leverage vs constraints.",
            "value_prop": "Converts capabilities into sharp value propositions; focuses on buyer personas, painful problems, and outcomes.",
            "gtm": "Designs go-to-market strategies; focuses on adoption sequencing, enterprise deployment, and organizational friction.",
            "railroad_intel": "Builds mental models of specific railroads; focuses on network structure, decision dynamics, and operational constraints.",
            "mna": "Thinks like a corporate development executive; identifies strategic buyers, synergies, and defensive value.",
            "artifact": "Converts inputs into polished artifacts; organizes and sharpens language without generating new strategy.",
        }

        self.agent_descriptions = "\n".join(
            [
                f"agent_id: {agent_id}\n description: {self.agent_descriptions_map[agent_id]}\n"
                for agent_id in self.agents
            ]
        )
        if not self.agent_descriptions:
            self.agent_descriptions = "No agents available for routing"
        logger.info(f"CSORouterAgent initialized with {len(self.agents)} agents")

    async def _run_classification(self, ctx: ChatContext, agent_descriptions: str) -> ChatAgent:
        prompt = classification_prompt.format(
            query=ctx.query,
            history=", ".join(message for message in ctx.history),
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
            )
            selected_agent_id = classification.agent_id if classification and classification.agent_id in self.agents else "strategy"
            logger.info(
                "CSORouterAgent selected '%s' with confidence %.2f",
                selected_agent_id,
                getattr(classification, "confidence_score", 0.0) or 0.0,
            )
            print("CSORouterAgent selected '%s' with confidence %.2f",
                selected_agent_id,
                getattr(classification, "confidence_score", 0.0) or 0.0,)
            
        except Exception as e:
            logger.error("Classification error, falling back to strategy agent: %s", e)
            selected_agent_id = "strategy"
        return self.agents[selected_agent_id]

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        return await agent.run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        async for chunk in agent.run_stream(ctx):
            yield chunk
