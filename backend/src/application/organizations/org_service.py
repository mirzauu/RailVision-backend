from src.infrastructure.database.repositories.org_repository import OrganizationRepository
from src.infrastructure.database.models import Organization

class OrgService:
    def __init__(self, org_repo: OrganizationRepository):
        self.org_repo = org_repo

    def create_organization(self, name: str) -> Organization:
        # Simple slug generation
        slug = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")
        # Ensure slug uniqueness logic here if needed, for now assume unique enough or catch integrity error
        # In a real app we'd append numbers if duplicate
        
        org = Organization(name=name, slug=slug)
        return self.org_repo.create(org)

    def get_organization(self, org_id: str) -> Organization:
        return self.org_repo.get_by_id(org_id)

    def get_organization_flexible(self, identifier: str) -> Organization | None:
        org = self.org_repo.get_by_id(identifier)
        if org:
            return org
        return self.org_repo.get_by_slug(identifier)
