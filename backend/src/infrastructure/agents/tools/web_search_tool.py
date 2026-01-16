import os
import asyncio
from typing import Any, Dict, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from src.infrastructure.llm.provider_service import ProviderService
from src.config.settings import settings


class WebSearchToolInput(BaseModel):
    query: str = Field(description="Query requiring online searching and reasoning.")


class WebSearchToolOutput(BaseModel):
    success: bool = Field(description="Whether answer generation was successful.")
    content: str = Field(description="Answer content.")
    citations: list[str] = Field(description="List of citation URLs.")


class WebSearchTool:
    name = "Web Search Tool"
    description = "Searches the internet for current and relevant information with citations."

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.api_key = os.getenv("PERPLEXITY_API_KEY") or os.getenv("LLM_API_KEY") or (settings.perplexity_api_key or "None")
        self.temperature = 0.3
        self.max_tokens = 12000
        self.output_schema = WebSearchToolOutput
        self.provider_service = ProviderService(self.user_id)

    async def arun(self, query: str) -> Dict[str, Any]:
        resp = await self._make_llm_call(query)
        if not resp:
            return {"success": False, "content": "Tool Call Failed", "citations": []}
        return resp

    async def _make_llm_call(self, query: str) -> Dict[str, Any]:
        try:
            messages = [{"role": "user", "content": query}]
            text_response = await self.providerm_service.call_llm_with_specific_model(
                model_identifier="perplexity/sonar",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            result = {
                "success": True,
                "content": text_response or "",
                "citations": [],
            }
            if isinstance(result.get("content"), str) and len(result["content"]) > 80000:
                result["content"] = result["content"][:80000]
            return result
        except Exception as e:
            return {"success": False, "content": f"LLM call failed: {str(e)}", "citations": []}


def web_search_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    tool_instance = WebSearchTool(sql_db, user_id)
    if tool_instance.api_key == "None":
        return None
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        name=tool_instance.name,
        description=tool_instance.description,
        args_schema=WebSearchToolInput,
    )
