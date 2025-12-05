from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolCallEventType(Enum):
    CALL = "call"
    RESULT = "result"


class ToolCallResponse(BaseModel):
    call_id: str = Field(..., description="ID of the tool call")
    event_type: ToolCallEventType = Field(..., description="Type of the event")
    tool_name: str = Field(..., description="Name of the tool")
    tool_response: str = Field(..., description="Response from the tool")
    tool_call_details: Dict[str, Any] = Field(..., description="Details of the tool call")


class ChatAgentResponse(BaseModel):
    response: str = Field(..., description="Full response to the query")
    tool_calls: List[ToolCallResponse] = Field([], description="List of tool calls")
    citations: List[str] = Field(..., description="List of file names referenced in the response")


class ChatContext(BaseModel):
    project_id: str
    history: List[str]
    query: str
    additional_context: str = ""


class TaskConfig(BaseModel):
    description: str
    expected_output: str
    context: Optional[Any] = None


class AgentConfig(BaseModel):
    role: str
    goal: str
    backstory: str
    tasks: List[TaskConfig]
    max_iter: int = 15


class ChatAgent(ABC):
    @abstractmethod
    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        pass

    @abstractmethod
    def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        pass

