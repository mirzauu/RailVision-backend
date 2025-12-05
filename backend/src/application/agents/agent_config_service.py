from src.infrastructure.database.repositories.agent_repository import AgentRepository
from src.infrastructure.database.models import Agent
from typing import List, Dict

class AgentConfigService:
    def __init__(self, agent_repo: AgentRepository):
        self.agent_repo = agent_repo

    def create_agent(self, name: str, type: str, config: Dict, org_id: str) -> Agent:
        agent = Agent(name=name, type=type, config=config, org_id=org_id)
        return self.agent_repo.create(agent)

    def get_agents_by_org(self, org_id: str) -> List[Agent]:
        return self.agent_repo.get_by_org(org_id)
    
    def get_agent(self, agent_id: str) -> Agent:
        return self.agent_repo.get_by_id(agent_id)
