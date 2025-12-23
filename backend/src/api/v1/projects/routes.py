from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.config.database import get_db
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User
from src.infrastructure.database.repositories.project_repository import ProjectRepository
from src.infrastructure.database.repositories.project_agent_repository import ProjectAgentRepository
from src.infrastructure.database.repositories.project_member_repository import ProjectMemberRepository

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

def get_project_service(db: Session = Depends(get_db)) -> ProjectService:
    return ProjectService(
        ProjectRepository(db),
        ProjectAgentRepository(db),
        ProjectMemberRepository(db),
    )

@router.post("/", response_model=ProjectResponse)
def create_project(
    project_in: ProjectCreate,
    svc: ProjectService = Depends(get_project_service),
    current_user: User = Depends(get_current_user),
):
    if not current_user.org_id:
        raise HTTPException(status_code=400, detail="User does not belong to an organization")
    return svc.create_project(
        org_id=current_user.org_id,
        created_by=current_user.id,
        name=project_in.name,
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
