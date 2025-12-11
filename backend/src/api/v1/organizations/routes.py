from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.config.database import get_db
from src.application.organizations.org_service import OrgService
from src.infrastructure.database.repositories.org_repository import OrganizationRepository
from src.infrastructure.database.repositories.role_repository import RoleRepository
from src.api.v1.organizations.schemas import OrgCreate, OrgResponse, OrgUserResponse
from typing import List
from src.infrastructure.database.repositories.user_repository import UserRepository
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User

router = APIRouter()

def get_org_service(db: Session = Depends(get_db)) -> OrgService:
    org_repo = OrganizationRepository(db)
    return OrgService(org_repo)

@router.post("/", response_model=OrgResponse)
def create_org(
    org_in: OrgCreate, 
    org_service: OrgService = Depends(get_org_service),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    org = org_service.create_organization(org_in.name)
    
    # Assign user to this org and make them admin
    # We need to fetch the admin role first
    role_repo = RoleRepository(db)
    admin_role = role_repo.get_by_name("org_admin")
    
    if not admin_role:
        # Fallback or error - strictly speaking seeds should have run. 
        # For now, let's assume it exists or raise error. 
        raise HTTPException(status_code=500, detail="System roles not initialized")

    current_user.org_id = org.id
    current_user.role_id = admin_role.id
    db.add(current_user)
    db.commit()
    
    return org

# Place static route before dynamic to avoid 'me' being captured by {identifier}
@router.get("/me", response_model=OrgResponse)
def get_my_org(
    org_service: OrgService = Depends(get_org_service),
    current_user: User = Depends(get_current_user)
):
    if not current_user.org_id:
        raise HTTPException(status_code=404, detail="Organization not found")
    org = org_service.get_organization(current_user.org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org

@router.get("/users", response_model=List[OrgUserResponse])
def list_org_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not current_user.org_id:
        return []
    users = UserRepository(db).get_by_org(current_user.org_id)
    return users

@router.get("/{identifier}", response_model=OrgResponse)
def get_org(
    identifier: str,
    org_service: OrgService = Depends(get_org_service),
    current_user: User = Depends(get_current_user)
):
    org = org_service.get_organization_flexible(identifier)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org
