from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from src.config.database import get_db
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User
from src.application.commercial.commercial_service import CommercialService
from src.api.v1.dashboard.commercial_schemas import CommercialMetricsResponse

router = APIRouter()

def get_commercial_service(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> CommercialService:
    return CommercialService(db, current_user.id)

@router.get("/", response_model=CommercialMetricsResponse)
def get_commercial_metrics(
    current_user: User = Depends(get_current_user),
    service: CommercialService = Depends(get_commercial_service)
):
    if not current_user.org_id:
        raise HTTPException(status_code=400, detail="User is not associated with an organization")
    
    return service.get_metrics(current_user.org_id)

@router.post("/refresh", response_model=CommercialMetricsResponse)
async def refresh_commercial_metrics(
    current_user: User = Depends(get_current_user),
    service: CommercialService = Depends(get_commercial_service)
):
    if not current_user.org_id:
        raise HTTPException(status_code=400, detail="User is not associated with an organization")
    
    try:
        return await service.refresh_metrics(current_user.org_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # In production, log the error
        raise HTTPException(status_code=500, detail="Internal server error")
