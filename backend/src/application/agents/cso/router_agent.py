import logging
from typing import AsyncGenerator, Dict

from pydantic import BaseModel, Field

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import ChatAgent, ChatAgentResponse, ChatContext
from .strategy_agent import CSOStrategyAgent
from .risk_agent import CSORiskAgent
from .execution_agent import CSOExecutionAgent


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
            "risk": CSORiskAgent(llm_provider),
            "execution": CSOExecutionAgent(llm_provider),
        }
        self.agent_descriptions_map: Dict[str, str] = {
            "strategy": "Analyzes the business model, value proposition, and competitive positioning; identifies growth levers, strategic bets, and long-term risks",
            "risk": "Evaluates business, operational, regulatory, and technical risks; assesses impact, likelihood, and mitigation options from an executive perspective",
            "execution": "Assesses execution feasibility, organizational readiness, key dependencies, and bottlenecks; flags strategy–execution gaps and prioritization issues",
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
            selected_agent_id = classification.agent_id if classification and classification.agent_id in self.agents else "explanation"
            logger.info(
                "CSORouterAgent selected '%s' with confidence %.2f",
                selected_agent_id,
                getattr(classification, "confidence_score", 0.0) or 0.0,
            )
            print("CSORouterAgent selected '%s' with confidence %.2f",
                selected_agent_id,
                getattr(classification, "confidence_score", 0.0) or 0.0,)
            
        except Exception as e:
            logger.error("Classification error, falling back to explanation agent: %s", e)
            selected_agent_id = "explanation"
        return self.agents[selected_agent_id]

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        return await agent.run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        async for chunk in agent.run_stream(ctx):
            yield chunk
