from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.config.database import get_db
from src.application.agents.agent_config_service import AgentConfigService
from src.infrastructure.database.repositories.agent_repository import AgentRepository
from src.api.v1.agents.schemas import AgentCreate, AgentResponse
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User
from typing import List

router = APIRouter()

def get_agent_service(db: Session = Depends(get_db)) -> AgentConfigService:
    agent_repo = AgentRepository(db)
    return AgentConfigService(agent_repo)

@router.post("/", response_model=AgentResponse)
def create_agent(
    agent_in: AgentCreate,
    agent_service: AgentConfigService = Depends(get_agent_service),
    current_user: User = Depends(get_current_user)
):
    if not current_user.org_id:
        raise HTTPException(status_code=400, detail="User does not belong to an organization")
        
    return agent_service.create_agent(
        name=agent_in.name,
        type=agent_in.type,
        config=agent_in.config,
        org_id=current_user.org_id
    )

@router.get("/", response_model=List[AgentResponse])
def list_agents(
    agent_service: AgentConfigService = Depends(get_agent_service),
    current_user: User = Depends(get_current_user)
):
    if not current_user.org_id:
        return []
    return agent_service.get_agents_by_org(current_user.org_id)
