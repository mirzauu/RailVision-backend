import logging
from fastapi import APIRouter, Depends, HTTPException

logger = logging.getLogger(__name__)
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from src.config.database import get_db
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User
from src.application.dashboard.dashboard_service import DashboardService
from src.application.commercial.commercial_service import CommercialService
from src.api.v1.dashboard.schemas import DashboardResponse

router = APIRouter()

def get_dashboard_service(db: Session = Depends(get_db)) -> DashboardService:
    return DashboardService(db)

@router.get("/", response_model=DashboardResponse)
def get_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    if not current_user.org_id:
        raise HTTPException(status_code=400, detail="User is not associated with an organization")
    
    # Update last active timestamp
    current_user.last_active_at = datetime.now(timezone.utc)
    db.add(current_user)
    db.commit()
        
    try:
        dashboard_data = dashboard_service.get_dashboard_data(current_user.org_id)
        
        # Inject commercial metrics
        try:
            commercial_service = CommercialService(db, current_user.id)
            commercial_metrics = commercial_service.get_metrics(current_user.org_id)
            dashboard_data.commercial = commercial_metrics
        except Exception as e:
            # Don't fail the whole dashboard if commercial data fails, just log it (in prod)
            # print(f"Failed to fetch commercial metrics: {e}")
            pass
            
        return dashboard_data
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching dashboard data")
