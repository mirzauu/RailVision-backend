import logging
from typing import AsyncGenerator, Dict, TYPE_CHECKING

from pydantic import BaseModel, Field

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import ChatAgent, ChatAgentResponse, ChatContext, AgentConfig, TaskConfig
from src.infrastructure.agents.pydantic_multi_agent import PydanticMultiAgent

if TYPE_CHECKING:
    from src.application.tools.service import ToolService

from .general_agent import CTOGeneralAgent
from .emily_agent import CTOEmilyAgent
from .innovation_agent import CTOInnovationAgent
from .architecture_agent import CTOArchitectureAgent
from .engineering_agent import CTOEngineeringAgent
from .security_agent import CTOSecurityAgent
from .infra_agent import CTOInfraAgent
from ..tool_agents.ppt_agent import CTOPPTAgent
from ..tool_agents.pdf_agent import CTOPDFAgent
from ..tool_agents.word_agent import CTOWordAgent
from ..tool_agents.spreadsheet_agent import CTOSpreadsheetAgent
from .brutall_agent import CTOBrutallAgent
from ..sub_agents.micheal_agent import CTOMichealAgent
from ..sub_agents.mary_agent import CTOMaryAgent
from ..sub_agents.rapheal_agent import CTORaphealAgent
from ..sub_agents.gabrial_agent import CTOGabrialAgent

logger = logging.getLogger(__name__)


class ClassificationResponse(BaseModel):
    agent_id: str = Field(description="agent_id of the best matching agent. Use 'multi_agent' if multiple agents are required.")
    confidence_score: float = Field(description="confidence score between 0 and 1")
    is_multi_agent: bool = Field(default=False, description="Set to True if the query requires coordination between multiple agents")


classification_prompt = (
    "You are part of the ai agentic system that routes the current query to the most appropriate CTO agent. "
    "Select the best agent by comparing the query's requirements with each agent's specialties.\n\n"
    "User Query: {query}\n"
    "Chat history: {history}\n"
    "--- end of Chat history ----\n\n"
    "Available agents and their specialties:\n"
    "{agent_descriptions}\n\n"
    "Analysis Instructions (do not include these in the final answer):\n"
    "1. Identify technical terms, system requirements, and the user's technology-related intent.\n"
    "2. Compare these elements to each agent's specialty description.\n"
    "3. Favor specialized agents over general ones for close matches.\n"
    "4. MULTI-AGENT REQUIRED: If the query requires expertise from multiple different technology domains or combining insights from more than one agent, explicitly select 'multi_agent' as the agent_id.\n"
)


CTO_SUPERVISOR_TASK_DESCRIPTION = """
You are Emily, the Head CTO of RailVision, operating in Multi-Agent Orchestration Mode.
You are the senior technology executive. You lead, synthesize, and take responsibility for the final technical output.

━━━━━━━━━━━━━━━━━━━━━━
STEP 1: UNDERSTAND THE TECHNOLOGY QUERY
━━━━━━━━━━━━━━━━━━━━━━

Before calling any subagent or tool, deeply understand the technical intent:
- Is this about Innovation (AI/R&D), Architecture (System Design), Engineering (Process), Security, or Infrastructure?
- Does it require combination of perspectives?
- Are there dependencies? (e.g., Infrastructure costs must be assessed before scaling Architecture)

━━━━━━━━━━━━━━━━━━━━━━
STEP 2: MANDATORY TODO TRACKING (FOR MULTI-TASK QUERIES)
━━━━━━━━━━━━━━━━━━━━━━

If the query involves multiple steps or deliverables, you MUST use the todo system to plan and track progress:
1. Use `create_todo` to create one todo per delegation or major step.
2. Use `update_todo_status` as you complete technical milestones.
3. Use `add_todo_note` to record architectural decisions or security risks.
4. Use `get_todo_summary` at the end to ensure the technical solution is fully structured before delivery.

━━━━━━━━━━━━━━━━━━━━━━
STEP 3: DELEGATE TO SUBAGENTS
━━━━━━━━━━━━━━━━━━━━━━

Use your delegate tools (consult_*_agent) to query specialists:
- `consult_emily_agent`: YOUR own strategic technical reasoning — use this when you need to think critically before synthesizing.
- `consult_innovation_agent`: Subagent for AI strategy, R&D roadmaps, and emerging tech.
- `consult_architecture_agent`: Subagent for system design, scalability, and technical platform R&D.
- `consult_engineering_agent`: Subagent for development velocity, code quality, and technical debt.
- `consult_security_agent`: Subagent for cybersecurity, data privacy, and risk governance.
- `consult_infra_agent`: Subagent for cloud operations, SRE, and infrastructure reliability.
- `consult_brutall_agent`: The Ruthless Mentor — use this to stress-test your technical architectures and engineering decisions.
- `consult_ppt_agent`: Generate technical PowerPoint presentations or slide decks.
- `consult_pdf_agent`: Generate polished technical PDF reports, whitepapers, or briefs.
- `consult_word_agent`: Generate structured technical Word documents or specifications.
- `consult_spreadsheet_agent`: Generate Excel spreadsheets for technical data or cost analysis.

━━━━━━━━━━━━━━━━━━━━━━
STEP 4: SYNTHESIZE & DELIVER
━━━━━━━━━━━━━━━━━━━━━━

Synthesize all technical insights into one cohesive executive-ready response:
- Lead with technical feasibility and long-term impact.
- Clearly distinguish: ✔ Verified Technical Facts | ~ Reasoned Inferences | ⚠ Technical Risks.
- Do NOT just concatenate subagent outputs — synthesize and challenge them.
"""


class CTORouterAgent(ChatAgent):
    def __init__(self, llm_provider: ProviderService, tools_provider: "ToolService"):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.agents: Dict[str, ChatAgent] = {
            "general": CTOGeneralAgent(llm_provider, tools_provider),
            "emily": CTOEmilyAgent(llm_provider, tools_provider),
            "innovation": CTOInnovationAgent(llm_provider, tools_provider),
            "architecture": CTOArchitectureAgent(llm_provider, tools_provider),
            "engineering": CTOEngineeringAgent(llm_provider, tools_provider),
            "security": CTOSecurityAgent(llm_provider, tools_provider),
            "infra": CTOInfraAgent(llm_provider, tools_provider),
            "ppt": CTOPPTAgent(llm_provider, tools_provider),
            "pdf": CTOPDFAgent(llm_provider, tools_provider),
            "word": CTOWordAgent(llm_provider, tools_provider),
            "spreadsheet": CTOSpreadsheetAgent(llm_provider, tools_provider),
            "brutall": CTOBrutallAgent(llm_provider, tools_provider),
            "micheal": CTOMichealAgent(llm_provider, tools_provider),
            "mary": CTOMaryAgent(llm_provider, tools_provider),
            "rapheal": CTORaphealAgent(llm_provider, tools_provider),
            "gabrial": CTOGabrialAgent(llm_provider, tools_provider),
        }
        self.agent_descriptions_map: Dict[str, str] = {
            "general": "Handles greetings and simple technology introductions for the CTO system.",
            "emily": "High-trust, senior CTO executive authority. Emily synthesizes complex technical challenges and provides strategic steering. Use for high-stakes technical advice.",
            "innovation": "Specializes in AI/ML strategy, technical roadmaps, sensor fusion research, and emerging technologies like edge-AI.",
            "architecture": "Specializes in scalable system design, cloud-to-edge communication architecture, and rigorous technical R&D.",
            "engineering": "Specializes in development velocity, engineering culture, CI/CD automation, and technical debt management.",
            "security": "Specializes in cybersecurity posture, data privacy compliance, and critical transport infrastructure protection.",
            "infra": "Specializes in site reliability (SRE), cloud operations, hosting costs, and devops automation.",
            "ppt": "Creates world-class, board-ready technical PowerPoint decks for RailVision's technology and engineering strategy. Use for ANY request to create, generate, or build a PowerPoint or .pptx presentation.",
            "pdf": "Produces consultant-grade technical PDF reports and whitepapers with deep analysis and professional design. Use for ANY request to create, generate, or build a PDF report.",
            "word": "Generates polished, deeply-researched technical Word (.docx) documents such as system specifications or engineering playbooks. Use for ANY request to create, generate, or build a Word document.",
            "spreadsheet": "Transforms structured engineering logs and technical cost requests into professional Excel (.xlsx) workbooks. Use for ANY request to create, generate, or build an Excel spreadsheet.",
            "brutall": "The Ruthless Mentor subagent. Use this to stress-test technical architectures and engineering decisions. Select when the user needs critical, 'no-holds-barred' technical feedback.",
            "micheal": "CSO Liaison subagent with deep knowledge of corporate strategy and enterprise value coordination.",
            "mary": "CCO Liaison subagent providing context on commercial reality, sales pipelines, and customer acquisition.",
            "rapheal": "CFO Liaison subagent focusing on financial metrics, budgeting, and capital allocation.",
            "gabrial": "CRO Liaison subagent focusing on revenue implications, target quotas, and sales performance.",
        }

        self.agent_descriptions = "\n".join(
            [
                f"agent_id: {agent_id}\n description: {self.agent_descriptions_map[agent_id]}\n"
                for agent_id in self.agents
            ]
        )
        
        self.supervisor_config = AgentConfig(
            role="Emily – Head CTO & Multi-Agent Orchestrator",
            goal="Coordinate specialized CTO subagents to deliver authoritative technical leadership.",
            backstory="You are Emily, the CTO of RailVision. You lead a team of elite technical specialists to ensure RailVision's technology is globally dominant, secure, and scalable.",
            tasks=[TaskConfig(description=CTO_SUPERVISOR_TASK_DESCRIPTION, expected_output="A cohesive, executive-ready technical advisory.")]
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
                "content": "You are an expert technical agent classifier that routes queries to the most appropriate CTO agent.",
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
                logger.info("CTORouterAgent selected 'multi_agent' mode")
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
                "CTORouterAgent selected '%s' with confidence %.2f",
                selected_agent_id,
                getattr(classification, "confidence_score", 0.0) or 0.0,
            )
            
        except Exception as e:
            logger.error("Classification error, falling back to general agent: %s", e)
            selected_agent_id = "general"
        return self.agents[selected_agent_id]

    def get_agent(self, agent_id: str) -> ChatAgent:
        return self.agents.get(agent_id) or self.agents["general"]

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        return await agent.run(ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        async for chunk in agent.run_stream(ctx):
            yield chunk
