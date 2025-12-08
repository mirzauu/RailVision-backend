import enum
from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, Text, ARRAY, DateTime
from sqlalchemy.types import JSON, Enum as AlchemyEnum
from sqlalchemy.orm import relationship

from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin

class IntegrationType(str, enum.Enum):
    GOOGLE_DRIVE = "google_drive"
    GOOGLE_WORKSPACE = "google_workspace"
    MICROSOFT_ONEDRIVE = "microsoft_onedrive"
    MICROSOFT_TEAMS = "microsoft_teams"
    SLACK = "slack"
    DISCORD = "discord"
    DROPBOX = "dropbox"
    BOX = "box"
    SALESFORCE = "salesforce"
    HUBSPOT = "hubspot"
    NOTION = "notion"
    CONFLUENCE = "confluence"
    JIRA = "jira"
    GITHUB = "github"
    GITLAB = "gitlab"
    CUSTOM_API = "custom_api"

class IntegrationStatus(str, enum.Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    AUTH_EXPIRED = "auth_expired"
    RATE_LIMITED = "rate_limited"

class Integration(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "integrations"

    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    
    type = Column(AlchemyEnum(IntegrationType, name="integration_type"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(AlchemyEnum(IntegrationStatus, name="integration_status"), default=IntegrationStatus.CONNECTED, nullable=False, index=True)
    
    access_token = Column(Text)
    refresh_token = Column(Text)
    token_expires_at = Column(DateTime(timezone=True))
    api_key = Column(Text)
    
    config = Column(JSON, default={})
    
    auto_sync_enabled = Column(Boolean, default=False)
    sync_frequency = Column(String(50), default='daily')
    last_sync_at = Column(DateTime(timezone=True))
    last_sync_status = Column(String(50))
    last_sync_error = Column(Text)
    next_sync_at = Column(DateTime(timezone=True), index=True)
    
    total_files_synced = Column(Integer, default=0)
    total_sync_runs = Column(Integer, default=0)
    failed_sync_count = Column(Integer, default=0)
    
    is_org_wide = Column(Boolean, default=True)
    project_ids = Column(JSON, default=[])
    
    connected_by = Column(String, ForeignKey("users.id"), nullable=False)
    
    metadata_ = Column("metadata", JSON, default={})

    organization = relationship("Organization", back_populates="integrations")
    connector = relationship("User", foreign_keys=[connected_by])
