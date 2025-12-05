from fastapi import APIRouter, Depends, Body
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import List, Optional
from src.config.database import get_db
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User
from src.infrastructure.llm.provider_service import ProviderService
from src.application.agents.base import AgentConfig, TaskConfig, ChatContext
from src.application.agents.executer_agent import ExecuterAgent

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    history: List[str] = Field(default_factory=list)
    project_id: str = "default"
    framework: str = "pydantic"
    role: str = "General Agent"
    goal: str = "Answer the query"
    backstory: str = "Helps with codebase questions"


@router.post("/chat")
async def chat(
    body: ChatRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    provider = ProviderService.create(user_id=str(current_user.id))
    config = AgentConfig(
        role=body.role,
        goal=body.goal,
        backstory=body.backstory,
        tasks=[
            TaskConfig(
                description="Answer the user's question using available context",
                expected_output="Clear, concise answer with any relevant references",
            )
        ],
    )
    agent = ExecuterAgent(provider, config, framework=body.framework)
    ctx = ChatContext(project_id=body.project_id or "default", history=body.history or [], query=body.query)
    resp = await agent.run(ctx)
    return {"response": resp.response}


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    provider = ProviderService.create(user_id=str(current_user.id))
    config = AgentConfig(
        role=body.role,
        goal=body.goal,
        backstory=body.backstory,
        tasks=[
            TaskConfig(
                description="Answer the user's question using available context",
                expected_output="Clear, concise answer with any relevant references",
            )
        ],
    )
    agent = ExecuterAgent(provider, config, framework=body.framework)
    ctx = ChatContext(project_id=body.project_id or "default", history=body.history or [], query=body.query)

    async def stream_response():
        async for chunk in agent.run_stream(ctx):
            yield json.dumps(chunk.model_dump()) + "\n"

    return StreamingResponse(stream_response(), media_type="application/json")
