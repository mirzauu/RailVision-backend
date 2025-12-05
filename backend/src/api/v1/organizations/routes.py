from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.config.database import get_db
from src.application.organizations.org_service import OrgService
from src.infrastructure.database.repositories.org_repository import OrganizationRepository
from src.api.v1.organizations.schemas import OrgCreate, OrgResponse
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
    current_user.org_id = org.id
    current_user.role = "admin"
    db.add(current_user)
    db.commit()
    
    return org

@router.get("/{org_id}", response_model=OrgResponse)
def get_org(
    org_id: str, 
    org_service: OrgService = Depends(get_org_service),
    current_user: User = Depends(get_current_user)
):
    org = org_service.get_organization(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org
