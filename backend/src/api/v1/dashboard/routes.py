from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.config.database import get_db
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User
from src.application.dashboard.dashboard_service import DashboardService
from src.api.v1.dashboard.schemas import DashboardResponse

router = APIRouter()

def get_dashboard_service(db: Session = Depends(get_db)) -> DashboardService:
    return DashboardService(db)

@router.get("/", response_model=DashboardResponse)
def get_dashboard(
    current_user: User = Depends(get_current_user),
    dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    if not current_user.org_id:
        raise HTTPException(status_code=400, detail="User is not associated with an organization")
        
    try:
        return dashboard_service.get_dashboard_data(current_user.org_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Log error in real app
        raise HTTPException(status_code=500, detail="Internal server error while fetching dashboard data")
