from sqlalchemy.orm import Session
from src.infrastructure.database.models import Agent
from typing import Optional, List

class AgentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, agent_id: str) -> Optional[Agent]:
        return self.db.query(Agent).filter(Agent.id == agent_id).first()

    def get_by_org(self, org_id: str) -> List[Agent]:
        return self.db.query(Agent).filter(Agent.org_id == org_id).all()

    def create(self, agent: Agent) -> Agent:
        self.db.add(agent)
        self.db.commit()
        self.db.refresh(agent)
        return agent
