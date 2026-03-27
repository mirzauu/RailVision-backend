from sqlalchemy.orm import Session
from sqlalchemy import func
from src.infrastructure.database.models import (
    Organization, User, Project, Agent, Document, Message, Conversation, LLMUsageLog
)
from src.api.v1.dashboard.schemas import DashboardResponse, QuotaInfo, DashboardStats, RecentItem
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

class DashboardService:
    def __init__(self, db: Session):
        self.db = db

    def get_dashboard_data(self, org_id: str) -> DashboardResponse:
        org = self.db.query(Organization).filter(Organization.id == org_id).first()
        if not org:
            raise ValueError("Organization not found")

        # Quotas
        user_count = self.db.query(func.count(User.id)).filter(User.org_id == org_id).scalar() or 0
        project_count = self.db.query(func.count(Project.id)).filter(Project.org_id == org_id).scalar() or 0
        agent_count = self.db.query(func.count(Agent.id)).filter(Agent.org_id == org_id).scalar() or 0
        doc_count = self.db.query(func.count(Document.id)).filter(Document.org_id == org_id).scalar() or 0
        
        storage_bytes = self.db.query(func.sum(Document.file_size_bytes)).filter(Document.org_id == org_id).scalar() or 0
        storage_gb = round(float(storage_bytes) / (1024**3), 2)
        
        # Monthly tokens (Simplified: all tokens for now, or could filter by current month)
        # Using LLMUsageLog for accurate token/cost tracking if available
        token_sum = self.db.query(func.sum(LLMUsageLog.total_tokens)).filter(LLMUsageLog.org_id == org_id).scalar() or 0
        
        quotas = {
            "users": self._to_quota(user_count, org.max_users),
            "projects": self._to_quota(project_count, org.max_projects),
            "agents": self._to_quota(agent_count, org.max_agents),
            "documents": self._to_quota(doc_count, org.max_documents),
            "storage": self._to_quota(storage_gb, org.max_storage_gb, unit="GB"),
            "tokens": self._to_quota(token_sum, org.max_monthly_tokens, unit="tokens")
        }

        # Stats
        total_convos = self.db.query(func.count(Conversation.id)).filter(Conversation.org_id == org_id).scalar() or 0
        total_msgs = self.db.query(func.count(Message.id)).filter(Message.org_id == org_id).scalar() or 0
        total_cost = self.db.query(func.sum(LLMUsageLog.cost_usd)).filter(LLMUsageLog.org_id == org_id).scalar() or Decimal("0")
        
        # Performance
        avg_res_time = self.db.query(func.avg(Agent.avg_response_time_ms)).filter(Agent.org_id == org_id).scalar()
        avg_satisfaction = self.db.query(func.avg(Agent.avg_satisfaction_rating)).filter(Agent.org_id == org_id).scalar()

        stats = DashboardStats(
            total_conversations=total_convos,
            total_messages=total_msgs,
            total_cost_usd=Decimal(str(total_cost)),
            avg_response_time_ms=float(avg_res_time) if avg_res_time else None,
            avg_satisfaction=float(avg_satisfaction) if avg_satisfaction else None
        )

        # Recent Activity
        recent_convos = (
            self.db.query(Conversation)
            .filter(Conversation.org_id == org_id)
            .order_by(Conversation.updated_at.desc())
            .limit(5)
            .all()
        )
        
        recent_projects = (
            self.db.query(Project)
            .filter(Project.org_id == org_id)
            .order_by(Project.updated_at.desc())
            .limit(5)
            .all()
        )
        
        recent_activity = []
        for c in recent_convos:
            recent_activity.append(RecentItem(
                id=str(c.id),
                name=c.title or "Untitled Conversation",
                type="conversation",
                updated_at=c.updated_at,
                status=c.status
            ))
        for p in recent_projects:
            recent_activity.append(RecentItem(
                id=str(p.id),
                name=p.name,
                type="project",
                updated_at=p.updated_at,
                status=p.status
            ))
            
        recent_activity.sort(key=lambda x: x.updated_at, reverse=True)

        return DashboardResponse(
            org_name=org.name,
            plan_type=org.plan_type,
            subscription_status=org.subscription_status,
            subscription_ends_at=org.subscription_ends_at,
            quotas=quotas,
            stats=stats,
            recent_activity=recent_activity[:10]
        )

    def _to_quota(self, used: float, max_val: Optional[float], unit: str = "count") -> QuotaInfo:
        m = max_val if max_val is not None else 0
        percentage = (used / m * 100) if m > 0 else 0
        return QuotaInfo(
            used=float(used),
            max=float(m),
            unit=unit,
            percentage=round(percentage, 2)
        )
