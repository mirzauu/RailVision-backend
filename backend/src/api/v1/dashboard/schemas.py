from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal
from src.api.v1.dashboard.commercial_schemas import CommercialMetricsResponse

class QuotaInfo(BaseModel):
    used: float
    max: float
    unit: str = "count"
    percentage: float

class DashboardStats(BaseModel):
    total_conversations: int
    total_messages: int
    total_cost_usd: Decimal
    avg_response_time_ms: Optional[float] = None
    avg_satisfaction: Optional[float] = None

class RecentItem(BaseModel):
    id: str
    name: str
    type: str # e.g., 'conversation', 'project'
    updated_at: datetime
    status: Optional[str] = None

class DashboardResponse(BaseModel):
    org_name: str
    plan_type: str
    subscription_status: str
    subscription_ends_at: Optional[datetime] = None
    
    quotas: Dict[str, QuotaInfo]
    stats: DashboardStats
    recent_activity: List[RecentItem]
    commercial: Optional[CommercialMetricsResponse] = None
    
    metadata: Dict[str, Any] = {}
