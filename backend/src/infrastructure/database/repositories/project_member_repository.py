from sqlalchemy.orm import Session
from typing import List
from src.infrastructure.database.models import ProjectMember

class ProjectMemberRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_project(self, project_id: str) -> List[ProjectMember]:
        return self.db.query(ProjectMember).filter(ProjectMember.project_id == project_id).all()

    def create(self, pm: ProjectMember) -> ProjectMember:
        self.db.add(pm)
        self.db.commit()
        self.db.refresh(pm)
        return pm

    def delete(self, pm: ProjectMember) -> None:
        self.db.delete(pm)
        self.db.commit()
