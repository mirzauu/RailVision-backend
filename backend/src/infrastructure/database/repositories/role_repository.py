from typing import Optional
from sqlalchemy.orm import Session
from src.infrastructure.database.models import Role

class RoleRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_name(self, name: str) -> Optional[Role]:
        return self.db.query(Role).filter(Role.name == name).first()
        
    def get_default_role(self) -> Optional[Role]:
        return self.db.query(Role).filter(Role.is_default == True).first()

    def create(self, role: Role) -> Role:
        self.db.add(role)
        self.db.commit()
        self.db.refresh(role)
        return role
