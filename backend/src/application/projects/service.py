from typing import Dict, Optional, List
from sqlalchemy.orm import Session
from src.infrastructure.database.models import Project, ProjectType, ProjectStatus, ProjectAgent, AgentRoleInProject, ProjectMember, MemberRoleInProject
from src.infrastructure.database.repositories.project_repository import ProjectRepository
from src.infrastructure.database.repositories.project_agent_repository import ProjectAgentRepository
from src.infrastructure.database.repositories.project_member_repository import ProjectMemberRepository
from src.infrastructure.database.repositories.agent_repository import AgentRepository

class ProjectService:
    def __init__(self, project_repo: ProjectRepository, project_agent_repo: ProjectAgentRepository, project_member_repo: ProjectMemberRepository, agent_repo: AgentRepository):
        self.project_repo = project_repo
        self.project_agent_repo = project_agent_repo
        self.project_member_repo = project_member_repo
        self.agent_repo = agent_repo

    def create_project(
        self,
        org_id: str,
        created_by: str,
        name: str,
        description: Optional[str] = None,
        type: Optional[str] = None,
        status: Optional[str] = None,
        settings: Optional[Dict] = None,
        objective: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        priority: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Project:
        pt = ProjectType(type) if type else ProjectType.SINGLE_CHAT
        ps = ProjectStatus(status) if status else ProjectStatus.ACTIVE
        p = Project(
            org_id=org_id,
            created_by=created_by,
            name=name,
            description=description,
            type=pt,
            status=ps,
            settings=settings or {},
            objective=objective,
            tags=tags or [],
            category=category,
            priority=priority or "medium",
        )
        p = self.project_repo.create(p)
        if agent_id:
            a = self.agent_repo.get_by_id(agent_id)
            if a and a.org_id == org_id:
                self.add_agent(p.id, agent_id, role=AgentRoleInProject.PRIMARY.value, assigned_by=created_by)
        return p

    def get_by_org(self, org_id: str) -> List[Project]:
        return self.project_repo.get_by_org(org_id)

    def get_by_id(self, project_id: str) -> Optional[Project]:
        return self.project_repo.get_by_id(project_id)

    def update_project(self, db: Session, project_id: str, data: Dict) -> Optional[Project]:
        p = self.project_repo.get_by_id(project_id)
        if not p:
            return None
        if "name" in data:
            p.name = data["name"]
        if "description" in data:
            p.description = data["description"]
        if "status" in data:
            p.status = ProjectStatus(data["status"]) if data["status"] else p.status
        if "type" in data:
            p.type = ProjectType(data["type"]) if data["type"] else p.type
        if "settings" in data:
            p.settings = data["settings"] or {}
        if "objective" in data:
            p.objective = data["objective"]
        if "tags" in data:
            p.tags = data["tags"] or []
        if "category" in data:
            p.category = data["category"]
        if "priority" in data:
            p.priority = data["priority"] or p.priority
        return self.project_repo.update(p)

    def add_agent(
        self,
        project_id: str,
        agent_id: str,
        role: Optional[str] = None,
        assigned_by: Optional[str] = None,
        project_config: Optional[Dict] = None,
    ) -> ProjectAgent:
        r = AgentRoleInProject(role) if role else AgentRoleInProject.PRIMARY
        pa = ProjectAgent(
            project_id=project_id,
            agent_id=agent_id,
            role=r,
            assigned_by=assigned_by,
            project_config=project_config or {},
        )
        return self.project_agent_repo.create(pa)

    def add_member(
        self,
        project_id: str,
        user_id: str,
        role: Optional[str] = None,
        invited_by: Optional[str] = None,
        notification_preferences: Optional[Dict] = None,
        permissions: Optional[Dict] = None,
    ) -> ProjectMember:
        r = MemberRoleInProject(role) if role else MemberRoleInProject.MEMBER
        pm = ProjectMember(
            project_id=project_id,
            user_id=user_id,
            role=r,
            invited_by=invited_by,
            notification_preferences=notification_preferences or {},
        )
        if permissions:
            if "can_invite_members" in permissions:
                pm.can_invite_members = bool(permissions["can_invite_members"])
            if "can_manage_agents" in permissions:
                pm.can_manage_agents = bool(permissions["can_manage_agents"])
            if "can_upload_documents" in permissions:
                pm.can_upload_documents = bool(permissions["can_upload_documents"])
            if "can_export_conversations" in permissions:
                pm.can_export_conversations = bool(permissions["can_export_conversations"])
        return self.project_member_repo.create(pm)

    def list_agents(self, project_id: str) -> List[ProjectAgent]:
        return self.project_agent_repo.get_by_project(project_id)

    def list_members(self, project_id: str) -> List[ProjectMember]:
        return self.project_member_repo.get_by_project(project_id)

    def get_projects_by_agent_for_user(self, agent_id: str, user_id: str, org_id: str) -> List[Project]:
        return self.project_repo.get_by_agent_and_user(agent_id, user_id, org_id)
