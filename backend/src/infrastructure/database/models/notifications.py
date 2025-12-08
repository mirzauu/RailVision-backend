import enum
from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, Text, ARRAY, DateTime
from sqlalchemy.types import JSON, Enum as AlchemyEnum
from sqlalchemy.orm import relationship

from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin

class NotificationType(str, enum.Enum):
    MENTION = "mention"
    REPLY = "reply"
    AGENT_RESPONSE = "agent_response"
    PROJECT_INVITE = "project_invite"
    DOCUMENT_SHARED = "document_shared"
    TASK_ASSIGNED = "task_assigned"
    PROJECT_UPDATE = "project_update"
    SYSTEM_ALERT = "system_alert"

class NotificationStatus(str, enum.Enum):
    UNREAD = "unread"
    READ = "read"
    ARCHIVED = "archived"

class Notification(Base, UUIDMixin):
    __tablename__ = "notifications"

    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    type = Column(AlchemyEnum(NotificationType, name="notification_type"), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    message = Column(Text, nullable=False)
    status = Column(AlchemyEnum(NotificationStatus, name="notification_status"), default=NotificationStatus.UNREAD, nullable=False, index=True)
    
    project_id = Column(String, ForeignKey("projects.id"))
    conversation_id = Column(String, ForeignKey("conversations.id"))
    message_id = Column(String, ForeignKey("messages.id"))
    document_id = Column(String, ForeignKey("documents.id"))
    
    action_url = Column(Text)
    action_label = Column(String(100))
    
    from_user_id = Column(String, ForeignKey("users.id"))
    from_agent_id = Column(String, ForeignKey("agents.id"))
    
    read_at = Column(DateTime(timezone=True))
    email_sent = Column(Boolean, default=False)
    email_sent_at = Column(DateTime(timezone=True))
    
    from sqlalchemy.sql import func
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    organization = relationship("Organization", back_populates="notifications")
    user = relationship("User", back_populates="notifications", foreign_keys=[user_id])
    
    # Optional relationships to source objects can be defined if needed, 
    # but might duplicate imports. Only define core ones.
