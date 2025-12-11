from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime, date
from src.infrastructure.database.models.projects import ProjectType, ProjectStatus, AgentRoleInProject, MemberRoleInProject

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    type: Optional[ProjectType] = None
    status: Optional[ProjectStatus] = None
    settings: Optional[Dict] = None
    objective: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    agent_id: Optional[str] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[ProjectType] = None
    status: Optional[ProjectStatus] = None
    settings: Optional[Dict] = None
    objective: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    priority: Optional[str] = None

class ProjectResponse(BaseModel):
    id: str
    org_id: str
    created_by: str
    name: str
    description: Optional[str]
    type: ProjectType
    status: ProjectStatus
    settings: Optional[Dict]
    objective: Optional[str]
    tags: Optional[List[str]]
    category: Optional[str]
    priority: Optional[str]
    created_at: datetime
    updated_at: datetime
    start_date: Optional[date]
    target_end_date: Optional[date]
    actual_end_date: Optional[date]

    class Config:
        from_attributes = True

class ProjectAgentCreate(BaseModel):
    agent_id: str
    role: Optional[AgentRoleInProject] = None
    project_config: Optional[Dict] = None

class ProjectMemberCreate(BaseModel):
    user_id: str
    role: Optional[MemberRoleInProject] = None
    notification_preferences: Optional[Dict] = None
    permissions: Optional[Dict] = None

class ProjectAgentResponse(BaseModel):
    id: str
    project_id: str
    agent_id: str
    role: AgentRoleInProject
    assigned_by: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class ProjectMemberResponse(BaseModel):
    id: str
    project_id: str
    user_id: str
    role: MemberRoleInProject
    invited_by: Optional[str]
    joined_at: datetime
    created_at: datetime

    class Config:
        from_attributes = True
