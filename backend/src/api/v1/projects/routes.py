from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from pathlib import Path

from src.config.database import get_db
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User
from src.infrastructure.database.repositories.project_repository import ProjectRepository
from src.infrastructure.database.repositories.project_agent_repository import ProjectAgentRepository
from src.infrastructure.database.repositories.project_member_repository import ProjectMemberRepository
from src.infrastructure.llm.provider_service import ProviderService

from src.application.projects.service import ProjectService
from src.api.v1.projects.schemas import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectAgentCreate,
    ProjectMemberCreate,
    ProjectAgentResponse,
    ProjectMemberResponse,
)

router = APIRouter()

class GeneratedProjectName(BaseModel):
    project_name: str = Field(description="The generated project name (2-5 words)")

def get_project_service(db: Session = Depends(get_db)) -> ProjectService:
    return ProjectService(
        ProjectRepository(db),
        ProjectAgentRepository(db),
        ProjectMemberRepository(db),
    )

async def generate_project_name(description: str, user_id: str) -> str:
    """Generate a creative project name using LLM based on description."""
    if not description:
        return "New Project"
        
    try:
        svc = ProviderService.create(user_id=user_id)
        
        # Load prompt template
        prompt_path = Path(__file__).parent / "prompts" / "name_generation.txt"
        if prompt_path.exists():
            prompt_template = prompt_path.read_text()
            system_content = prompt_template.replace("{{description}}", description)
        else:
            # Fallback if prompt file missing
            system_content = f"Generate a creative, short (2-5 words) project name based on this description: {description}. Return JSON with key 'project_name'."

        messages = [
            {"role": "system", "content": "You are a creative naming assistant. Output valid JSON only."},
            {"role": "user", "content": system_content}
        ]
        
        result = await svc.call_llm_with_structured_output(
            messages=messages,
            output_schema=GeneratedProjectName,
            config_type="chat"
        )
        
        return result.project_name
        
    except Exception as e:
        # Fallback in case of LLM failure
        import logging
        logging.error(f"Failed to generate project name: {e}")
        return "New Project"

@router.post("/", response_model=ProjectResponse)
async def create_project(
    project_in: ProjectCreate,
    svc: ProjectService = Depends(get_project_service),
    current_user: User = Depends(get_current_user),
):
    if not current_user.org_id:
        raise HTTPException(status_code=400, detail="User does not belong to an organization")
    
    # Generate name if description provided, otherwise use provided name or default
    final_name = project_in.name
    if project_in.description:
        final_name = await generate_project_name(project_in.description, current_user.id)
    
    return svc.create_project(
        org_id=current_user.org_id,
        created_by=current_user.id,
        name=final_name,
        description=project_in.description,
        type=project_in.type.value if project_in.type else None,
        status=project_in.status.value if project_in.status else None,
        settings=project_in.settings,
        objective=project_in.objective,
        tags=project_in.tags,
        category=project_in.category,
        priority=project_in.priority,
        agent_id=project_in.agent_id,
    )

@router.get("/", response_model=List[ProjectResponse])
def list_projects(
    svc: ProjectService = Depends(get_project_service),
    current_user: User = Depends(get_current_user),
):
    if not current_user.org_id:
        return []
    return svc.get_by_org(current_user.org_id)

@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: str,
    svc: ProjectService = Depends(get_project_service),
    current_user: User = Depends(get_current_user),
):
    p = svc.get_by_id(project_id)
    if not p or p.org_id != current_user.org_id:
        raise HTTPException(status_code=404, detail="Project not found")
    return p

@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: str,
    update: ProjectUpdate,
    db: Session = Depends(get_db),
    svc: ProjectService = Depends(get_project_service),
    current_user: User = Depends(get_current_user),
):
    p = svc.get_by_id(project_id)
    if not p or p.org_id != current_user.org_id:
        raise HTTPException(status_code=404, detail="Project not found")
    data = update.model_dump(exclude_unset=True)
    if "type" in data and data["type"] is not None:
        data["type"] = data["type"].value
    if "status" in data and data["status"] is not None:
        data["status"] = data["status"].value
    upd = svc.update_project(db, project_id, data)
    if not upd:
        raise HTTPException(status_code=404, detail="Project not found")
    return upd

@router.post("/{project_id}/agents", response_model=ProjectAgentResponse)
def add_agent(
    project_id: str,
    body: ProjectAgentCreate,
    svc: ProjectService = Depends(get_project_service),
    current_user: User = Depends(get_current_user),
):
    p = svc.get_by_id(project_id)
    if not p or p.org_id != current_user.org_id:
        raise HTTPException(status_code=404, detail="Project not found")
    role = body.role.value if body.role else None
    return svc.add_agent(project_id, body.agent_id, role, assigned_by=current_user.id, project_config=body.project_config)

@router.get("/{project_id}/agents", response_model=List[ProjectAgentResponse])
def list_agents(
    project_id: str,
    svc: ProjectService = Depends(get_project_service),
    current_user: User = Depends(get_current_user),
):
    p = svc.get_by_id(project_id)
    if not p or p.org_id != current_user.org_id:
        raise HTTPException(status_code=404, detail="Project not found")
    return svc.list_agents(project_id)

@router.post("/{project_id}/members", response_model=ProjectMemberResponse)
def add_member(
    project_id: str,
    body: ProjectMemberCreate,
    svc: ProjectService = Depends(get_project_service),
    current_user: User = Depends(get_current_user),
):
    p = svc.get_by_id(project_id)
    if not p or p.org_id != current_user.org_id:
        raise HTTPException(status_code=404, detail="Project not found")
    role = body.role.value if body.role else None
    return svc.add_member(
        project_id,
        body.user_id,
        role,
        invited_by=current_user.id,
        notification_preferences=body.notification_preferences,
        permissions=body.permissions,
    )

@router.get("/{project_id}/members", response_model=List[ProjectMemberResponse])
def list_members(
    project_id: str,
    svc: ProjectService = Depends(get_project_service),
    current_user: User = Depends(get_current_user),
):
    p = svc.get_by_id(project_id)
    if not p or p.org_id != current_user.org_id:
        raise HTTPException(status_code=404, detail="Project not found")
    return svc.list_members(project_id)

@router.get("/by-agent/{agent_id}", response_model=List[ProjectResponse])
def list_projects_by_agent(
    agent_id: str,
    svc: ProjectService = Depends(get_project_service),
    current_user: User = Depends(get_current_user),
):
    if not current_user.org_id:
        return []
    return svc.get_projects_by_agent_for_user(agent_id, current_user.id, current_user.org_id)
