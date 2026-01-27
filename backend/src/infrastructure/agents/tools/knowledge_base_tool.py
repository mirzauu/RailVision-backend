import asyncio
from typing import Any, Dict, List, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from src.application.reasoning.pipeline import context_enrich


class KnowledgeBaseToolInput(BaseModel):
    query: str = Field(description="The question or query to search for in the internal knowledge base.")


class KnowledgeBaseTool:
    """Tool for retrieving information from the internal knowledge base (Pinecone and Neo4j)."""

    name = "knowledge_base"
    description = """
Use this tool to retrieve strategic facts and supporting context from the internal knowledge base.
The knowledge base contains:
- Strategic Facts (Neo4j): Authoritative data on products, markets, capabilities, risks, goals, etc.
- Supporting Context (Pinecone): Explanatory documentation and details extracted from uploaded files.

This tool is essential for answering questions about the company's internal data, strategies, 
and relationships between various business entities.
"""

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id

    async def arun(self, query: str) -> str:
        """Retrieve information from the knowledge base asynchronously."""
        try:
            # Call the context_enrich pipeline which handles intent classification,
            # Pinecone retrieval, and Neo4j state building.
            result = await context_enrich(
                question=query,
                user_id=self.user_id
            )
            return result
        except Exception as e:
            return f"Error retrieving knowledge base info: {str(e)}"

    def run(self, query: str) -> str:
        """Synchronous wrapper for arun."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Handle case where loop is already running
        if loop.is_running():
            # In most modern async environments, this might be tricky,
            # but StructuredTool from_function handles async coroutines directly.
            # This sync run is mostly for compatibility.
            return "Error: Cannot run knowledge_base tool synchronously in this environment."

        try:
            return loop.run_until_complete(self.arun(query))
        finally:
            pass


def knowledge_base_tool(sql_db: Session, user_id: str) -> StructuredTool:
    """Create and return the knowledge base tool."""
    tool_instance = KnowledgeBaseTool(sql_db, user_id)
    return StructuredTool.from_function(
        func=tool_instance.run,
        coroutine=tool_instance.arun,
        name=tool_instance.name,
        description=tool_instance.description,
        args_schema=KnowledgeBaseToolInput,
    )
