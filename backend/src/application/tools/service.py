from typing import List, Dict
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from src.infrastructure.agents.tools.think_tool import think_tool
from src.infrastructure.agents.tools.web_search_tool import web_search_tool
from src.infrastructure.agents.tools.knowledge_base_tool import knowledge_base_tool
from src.infrastructure.agents.tools.ppt_tool import ppt_generation_tool
from src.infrastructure.agents.tools.pdf_tool import pdf_generation_tool
from src.infrastructure.agents.tools.attachment_tool import attachment_search_tool
from src.api.v1.tools.schemas import ToolInfo, ToolInfoWithParameters
from src.infrastructure.llm.provider_service import ProviderService



class ToolService:
    def __init__(self, db: Session, user_id: str, conversation_id: str | None = None):
        self.db = db
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize tools properties (placeholders for now)
        self.webpage_extractor_tool = None # webpage_extractor_tool(db, user_id)
        self.web_search_tool = None # web_search_tool(db, user_id)
        self.github_tool = None # github_tool(db, user_id)
        
        self.provider_service = ProviderService(user_id)
        self.tools = self._initialize_tools()

    def _initialize_tools(self) -> Dict[str, StructuredTool]:
        tools = {
            "think": think_tool(self.db, self.user_id),
        }
        ws = web_search_tool(self.db, self.user_id)
        if ws:
            tools["web_search_tool"] = ws

        tools["knowledge_base"] = knowledge_base_tool(self.db, self.user_id)

        if self.webpage_extractor_tool:
            tools["webpage_extractor"] = self.webpage_extractor_tool

        if self.github_tool:
            tools["github_tool"] = self.github_tool

        if self.web_search_tool:
            tools["web_search_tool"] = self.web_search_tool

        # PPT Generation Tools
        ppt_tools = ppt_generation_tool(self.db, self.user_id, self.conversation_id)
        for ppt_tool in ppt_tools:
            tools[ppt_tool.name] = ppt_tool

        # PDF Generation Tools
        pdf_tools = pdf_generation_tool(self.db, self.user_id, self.conversation_id)
        if pdf_tools:
            for pdf_tool in pdf_tools:
                tools[pdf_tool.name] = pdf_tool

        # Attachment Search Tool
        att_tool = attachment_search_tool(self.db, self.user_id, self.conversation_id)
        if att_tool:
            tools[att_tool.name] = att_tool

        return tools

    def get_tools(self, tool_names: List[str]) -> List[StructuredTool]:
        """get tools if exists"""
        tools = []
        for tool_name in tool_names:
            if self.tools.get(tool_name) is not None:
                tools.append(self.tools[tool_name])
        return tools

    def list_tools(self) -> List[ToolInfo]:
        return [
            ToolInfo(
                id=tool_id,
                name=tool.name,
                description=tool.description,
            )
            for tool_id, tool in self.tools.items()
        ]

    def list_tools_with_parameters(self) -> Dict[str, ToolInfoWithParameters]:
        return {
            tool_id: ToolInfoWithParameters(
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_schema.schema() if tool.args_schema else {},
            )
            for tool_id, tool in self.tools.items()
        }
