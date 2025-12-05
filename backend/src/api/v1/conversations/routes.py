from fastapi import APIRouter, Depends
from pydantic import BaseModel
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
    history: Optional[List[str]] = []
    project_id: Optional[str] = "default"
    framework: Optional[str] = "pydantic"
    role: Optional[str] = "General Agent"
    goal: Optional[str] = "Answer the query"
    backstory: Optional[str] = "Helps with codebase questions"


@router.post("/chat")
async def chat(
    body: ChatRequest,
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
