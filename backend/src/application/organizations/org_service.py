from src.infrastructure.database.repositories.org_repository import OrganizationRepository
from src.infrastructure.database.models import Organization

class OrgService:
    def __init__(self, org_repo: OrganizationRepository):
        self.org_repo = org_repo

    def create_organization(self, name: str) -> Organization:
        org = Organization(name=name)
        return self.org_repo.create(org)

    def get_organization(self, org_id: str) -> Organization:
        return self.org_repo.get_by_id(org_id)
