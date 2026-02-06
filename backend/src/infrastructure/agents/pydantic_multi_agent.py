import logging
import re
from typing import List, Dict, Optional, Any, AsyncGenerator

from langchain_core.tools import StructuredTool
from pydantic_ai import Tool, RunContext

from src.infrastructure.llm.provider_service import ProviderService
from src.domain.agents.base import (
    AgentConfig,
    ChatAgent,
    ChatAgentResponse,
    ChatContext,
    ToolCallEventType,
    ToolCallResponse,
)
from src.domain.agents.agent_types import AgentType
from .pydantic_agent import PydanticChatAgent

logger = logging.getLogger(__name__)

class PydanticMultiAgent(PydanticChatAgent):
    """
    Multi-agent system using Pydantic AI with agent delegation patterns.
    Extends PydanticChatAgent to serve as a supervisor that can delegate tasks to other agents.
    """

    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[StructuredTool] | None = None,
        mcp_servers: List[dict] | None = None,
        delegate_agents: Optional[Dict[str, AgentConfig]] = None,
        existing_delegates: Optional[Dict[str, ChatAgent]] = None,
        delegate_descriptions: Optional[Dict[str, str]] = None,
        tools_provider: Any = None,
    ):
        """
        Initialize the multi-agent system.
        
        Args:
            llm_provider: The LLM provider service
            config: Agent configuration (for the supervisor)
            tools: List of tools to use
            mcp_servers: Optional MCP servers configuration (kept for signature compatibility)
            delegate_agents: Optional delegate agent configurations
            existing_delegates: Optional dict of already initialized agents
            delegate_descriptions: Optional dict of descriptions for existing delegates
            tools_provider: Optional ToolService (kept for signature compatibility)
        """
        self.delegate_configs = delegate_agents or {}
        self.delegates: Dict[str, ChatAgent] = {}
        
        # Add existing delegates if provided
        if existing_delegates:
            self.delegates.update(existing_delegates)
            
        delegate_tools = []
        
        # Initialize delegate agents from configs and their corresponding tools
        for agent_type_key, agent_config in self.delegate_configs.items():
            # Use string representation of AgentType if it's an Enum
            agent_type_str = str(agent_type_key.value) if hasattr(agent_type_key, "value") else str(agent_type_key)
            
            # Initialize the delegate agent
            # We pass empty tools to delegates for now, assuming they are pure reasoning agents
            # or their tools are configured in the config object (if supported later)
            delegate = PydanticChatAgent(llm_provider, agent_config, tools=[])
            self.delegates[agent_type_str] = delegate
            
            # Create a tool that allows the supervisor to call this agent
            tool = self._create_delegate_tool(agent_type_str, agent_config)
            delegate_tools.append(tool)
            
        # Create tools for existing delegates
        if existing_delegates and delegate_descriptions:
            for name, description in delegate_descriptions.items():
                if name in self.delegates:
                    tool = self._create_existing_delegate_tool(name, description)
                    delegate_tools.append(tool)
            
        # Combine provided tools with delegate tools
        # We allow tools to be None, matching PydanticChatAgent signature logic
        combined_tools = (tools or []) + delegate_tools
        
        # Initialize the supervisor (self) via PydanticChatAgent constructor
        super().__init__(llm_provider, config, tools=combined_tools)
        
        logger.info(f"Initialized PydanticMultiAgent with delegates: {list(self.delegates.keys())}")

    def _create_existing_delegate_tool(self, agent_name: str, description: str) -> Tool:
        """
        Creates a Tool instance for an existing delegate agent.
        """
        tool_name = f"consult_{agent_name.lower().replace(' ', '_')}_agent"
        full_description = (
            f"Delegate a task to the {agent_name} agent. "
            f"Description: {description}. "
            f"Use this tool when the query requires expertise in {agent_name} domain."
        )

        async def delegate_task(ctx: RunContext[Any], query: str) -> str:
            """
            Delegates the query to the specific agent and returns their response.
            """
            logger.info(f"Supervisor delegating to {agent_name} agent: {query}")
            
            agent = self.delegates.get(agent_name)
            if not agent:
                return f"Error: Agent {agent_name} not found."
                
            # Create a context for the delegate
            delegate_ctx = ChatContext(
                query=query,
                history=[],  # Start fresh for the specific delegated task
                conversation_id=f"delegate-{agent_name}",
                user_id="system-delegation"
            )
            
            try:
                response: ChatAgentResponse = await agent.run(delegate_ctx)
                logger.info(f"Delegate {agent_name} finished.")
                return response.response
            except Exception as e:
                logger.error(f"Error executing delegate {agent_name}: {e}")
                return f"Error executing {agent_name} agent: {str(e)}"

        # Clean tool name as per PydanticAI expectation (no spaces)
        clean_name = re.sub(r" ", "", tool_name)
        
        return Tool(delegate_task, name=clean_name, description=full_description)

    def _create_delegate_tool(self, agent_type: str, config: AgentConfig) -> Tool:
        """
        Creates a Tool instance that delegates execution to a sub-agent.
        """
        tool_name = f"consult_{agent_type.lower().replace(' ', '_')}_agent"
        description = (
            f"Delegate a task to the {agent_type} agent. "
            f"Role: {config.role}. "
            f"Goal: {config.goal}. "
            f"Use this tool when the query requires expertise in {agent_type} domain."
        )

        async def delegate_task(ctx: RunContext[Any], query: str) -> str:
            """
            Delegates the query to the specific agent and returns their response.
            """
            logger.info(f"Supervisor delegating to {agent_type} agent: {query}")
            
            agent = self.delegates.get(agent_type)
            if not agent:
                return f"Error: Agent {agent_type} not found."
                
            # Create a context for the delegate
            # We create a fresh context treating the 'query' as the user's input to the delegate
            delegate_ctx = ChatContext(
                query=query,
                history=[],  # Start fresh for the specific delegated task
                conversation_id=f"delegate-{agent_type}",
                user_id="system-delegation"
            )
            
            try:
                response: ChatAgentResponse = await agent.run(delegate_ctx)
                logger.info(f"Delegate {agent_type} finished.")
                return response.response
            except Exception as e:
                logger.error(f"Error executing delegate {agent_type}: {e}")
                return f"Error executing {agent_type} agent: {str(e)}"

        # Clean tool name as per PydanticAI expectation (no spaces)
        clean_name = re.sub(r" ", "", tool_name)
        
        return Tool(delegate_task, name=clean_name, description=description)
