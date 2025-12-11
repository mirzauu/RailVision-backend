from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import Optional, List
from src.infrastructure.database.models import Project, ProjectAgent, ProjectMember

class ProjectRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, project_id: str) -> Optional[Project]:
        return self.db.query(Project).filter(Project.id == project_id).first()

    def get_by_org(self, org_id: str) -> List[Project]:
        return self.db.query(Project).filter(Project.org_id == org_id).all()

    def create(self, project: Project) -> Project:
        self.db.add(project)
        self.db.commit()
        self.db.refresh(project)
        return project

    def update(self, project: Project) -> Project:
        self.db.commit()
        self.db.refresh(project)
        return project

    def get_by_agent_and_user(self, agent_id: str, user_id: str, org_id: str) -> List[Project]:
        q = (
            self.db.query(Project)
            .join(ProjectAgent, ProjectAgent.project_id == Project.id)
            .outerjoin(ProjectMember, ProjectMember.project_id == Project.id)
            .filter(
                Project.org_id == org_id,
                ProjectAgent.agent_id == agent_id,
                or_(ProjectMember.user_id == user_id, Project.created_by == user_id),
            )
            .distinct(Project.id)
        )
        return q.all()
