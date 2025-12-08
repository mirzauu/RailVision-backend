import enum
from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, Text, BigInteger, DateTime, Numeric
from sqlalchemy.sql import func
from sqlalchemy.types import JSON, Enum as AlchemyEnum
from sqlalchemy.orm import relationship

from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin

class LLMUsageLog(Base):
    __tablename__ = "llm_usage_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(String, ForeignKey("projects.id"), index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    agent_id = Column(String, ForeignKey("agents.id"), index=True)
    
    conversation_id = Column(String, ForeignKey("conversations.id"), index=True)
    message_id = Column(String, ForeignKey("messages.id"))
    
    provider = Column(String(50), nullable=False)
    model = Column(String(100), nullable=False, index=True)
    endpoint = Column(String(100))
    
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer, nullable=False)
    
    cost_usd = Column(Numeric(10, 6))
    
    request_duration_ms = Column(Integer)
    time_to_first_token_ms = Column(Integer)
    tokens_per_second = Column(Numeric(10, 2))
    
    status = Column(String(50), nullable=False)
    error_code = Column(String(100))
    error_message = Column(Text)
    
    rate_limit_remaining = Column(Integer)
    rate_limit_reset_at = Column(DateTime(timezone=True))
    
    request_hash = Column(String(64))
    
    metadata_ = Column("metadata", JSON, default={})
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

class AuditAction(str, enum.Enum):
    # Auth
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_LOGIN_FAILED = "user_login_failed"
    PASSWORD_CHANGED = "password_changed"
    
    # Users
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    USER_INVITED = "user_invited"
    ROLE_CHANGED = "role_changed"
    
    # Organizations
    ORG_CREATED = "org_created"
    ORG_UPDATED = "org_updated"
    ORG_SETTINGS_CHANGED = "org_settings_changed"
    SUBSCRIPTION_CHANGED = "subscription_changed"
    
    # Projects
    PROJECT_CREATED = "project_created"
    PROJECT_UPDATED = "project_updated"
    PROJECT_DELETED = "project_deleted"
    PROJECT_ARCHIVED = "project_archived"
    PROJECT_MEMBER_ADDED = "project_member_added"
    PROJECT_MEMBER_REMOVED = "project_member_removed"
    PROJECT_AGENT_ASSIGNED = "project_agent_assigned"
    
    # Agents
    AGENT_CREATED = "agent_created"
    AGENT_UPDATED = "agent_updated"
    AGENT_DELETED = "agent_deleted"
    AGENT_CONFIG_CHANGED = "agent_config_changed"
    AGENT_EXECUTED = "agent_executed"
    SUB_AGENT_CREATED = "sub_agent_created"
    
    # Documents
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_VIEWED = "document_viewed"
    DOCUMENT_DOWNLOADED = "document_downloaded"
    DOCUMENT_DELETED = "document_deleted"
    DOCUMENT_SHARED = "document_shared"
    DOCUMENT_INGESTED = "document_ingested"
    
    # Conversations
    CONVERSATION_STARTED = "conversation_started"
    MESSAGE_SENT = "message_sent"
    CONVERSATION_ARCHIVED = "conversation_archived"
    
    # Integrations
    INTEGRATION_CONNECTED = "integration_connected"
    INTEGRATION_DISCONNECTED = "integration_disconnected"
    INTEGRATION_SYNCED = "integration_synced"

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(String, ForeignKey("projects.id"), index=True)
    
    user_id = Column(String, ForeignKey("users.id"), index=True)
    
    action = Column(AlchemyEnum(AuditAction, name="audit_action"), nullable=False, index=True)
    resource_type = Column(String(100), index=True)
    resource_id = Column(String, index=True)
    
    details = Column(JSON, default={})
    
    ip_address = Column(String) # INET in postgres
    user_agent = Column(Text)
    request_id = Column(String)
    
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

class CollaborationType(str, enum.Enum):
    DELEGATION = "delegation"
    CONSULTATION = "consultation"
    PARALLEL_EXECUTION = "parallel_execution"
    SEQUENTIAL_WORKFLOW = "sequential_workflow"

class AgentCollaborationLog(Base, UUIDMixin):
    __tablename__ = "agent_collaboration_logs"

    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    project_id = Column(String, ForeignKey("projects.id"))
    conversation_id = Column(String, ForeignKey("conversations.id"), index=True)
    message_id = Column(String, ForeignKey("messages.id"))
    
    type = Column(AlchemyEnum(CollaborationType, name="collaboration_type"), nullable=False)
    primary_agent_id = Column(String, ForeignKey("agents.id"), nullable=False, index=True)
    collaborating_agent_id = Column(String, ForeignKey("agents.id"), nullable=False, index=True)
    
    task_description = Column(Text)
    input_data = Column(JSON)
    output_data = Column(JSON)
    
    duration_ms = Column(Integer)
    tokens_used = Column(Integer)
    cost_usd = Column(Numeric(10, 6))
    
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
