from sqlalchemy.orm import Session
from typing import List
from src.infrastructure.database.models import ProjectAgent

class ProjectAgentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_project(self, project_id: str) -> List[ProjectAgent]:
        return self.db.query(ProjectAgent).filter(ProjectAgent.project_id == project_id).all()

    def create(self, pa: ProjectAgent) -> ProjectAgent:
        self.db.add(pa)
        self.db.commit()
        self.db.refresh(pa)
        return pa

    def delete(self, pa: ProjectAgent) -> None:
        self.db.delete(pa)
        self.db.commit()
