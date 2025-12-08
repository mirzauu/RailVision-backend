import enum
from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, Text, ARRAY, Date, DateTime
from sqlalchemy.sql import func
from sqlalchemy.types import JSON, Enum as AlchemyEnum
from sqlalchemy.orm import relationship

from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin

class ProjectType(str, enum.Enum):
    SINGLE_CHAT = "single_chat"
    GROUP_CHAT = "group_chat"
    WORKFLOW = "workflow"
    ANALYSIS = "analysis"

class ProjectStatus(str, enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    PAUSED = "paused"
    COMPLETED = "completed"

class AgentRoleInProject(str, enum.Enum):
    PRIMARY = "primary"
    SUPPORTING = "supporting"
    ADVISOR = "advisor"
    OBSERVER = "observer"

class MemberRoleInProject(str, enum.Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    GUEST = "guest"
    OBSERVER = "observer"

class Project(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "projects"

    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    name = Column(String(255), nullable=False)
    description = Column(Text)
    type = Column(AlchemyEnum(ProjectType, name="project_type"), default=ProjectType.SINGLE_CHAT, nullable=False, index=True)
    status = Column(AlchemyEnum(ProjectStatus, name="project_status"), default=ProjectStatus.ACTIVE, nullable=False, index=True)
    
    settings = Column(JSON, default={})
    
    objective = Column(Text)
    deliverables = Column(JSON, default=[])
    
    start_date = Column(Date)
    target_end_date = Column(Date)
    actual_end_date = Column(Date)
    
    last_activity_at = Column(DateTime(timezone=True), index=True)
    message_count = Column(Integer, default=0)
    
    tags = Column(JSON, default=[], index=True)
    category = Column(String(100))
    priority = Column(String(50), default='medium')
    
    metadata_ = Column("metadata", JSON, default={})

    # Relationships
    organization = relationship("Organization", back_populates="projects")
    creator = relationship("User", back_populates="projects_created", foreign_keys=[created_by])
    
    agent_associations = relationship("ProjectAgent", back_populates="project", cascade="all, delete-orphan")
    members = relationship("ProjectMember", back_populates="project", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="project", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="project") # Document can optionally belong to a project

class ProjectAgent(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "project_agents"

    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_id = Column(String, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    role = Column(AlchemyEnum(AgentRoleInProject, name="agent_role_in_project"), default=AgentRoleInProject.PRIMARY, nullable=False, index=True)
    
    can_initiate_conversation = Column(Boolean, default=True)
    can_view_all_conversations = Column(Boolean, default=True)
    can_access_project_documents = Column(Boolean, default=True)
    
    project_config = Column(JSON, default={})
    
    messages_sent = Column(Integer, default=0)
    last_active_at = Column(DateTime(timezone=True))
    
    assigned_by = Column(String, ForeignKey("users.id"))
    assigned_at = Column(DateTime(timezone=True), server_default=func.now())
    # The requirement asks for assigned_at. created_at from mixin effectively covers this, but let's keep it explicit if needed or just use created_at.
    # The DB schema has assigned_at distinct from created_at in some contexts, but here it says `assigned_at ... created_at`. They are redundant if assigned == created.
    # `assigned_at` is NOT in the Mixin. I will assume created_at suffices or map it.
    # Actually, let's just use the fields appearing in the schema.
    
    # assignments
    project = relationship("Project", back_populates="agent_associations")
    agent = relationship("Agent", back_populates="project_assignments")
    # assigner = relationship("User", foreign_keys=[assigned_by])

class ProjectMember(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "project_members"

    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    role = Column(AlchemyEnum(MemberRoleInProject, name="member_role_in_project"), default=MemberRoleInProject.MEMBER, nullable=False, index=True)
    
    can_invite_members = Column(Boolean, default=False)
    can_manage_agents = Column(Boolean, default=False)
    can_upload_documents = Column(Boolean, default=True)
    can_export_conversations = Column(Boolean, default=True)
    
    messages_sent = Column(Integer, default=0)
    last_viewed_at = Column(DateTime(timezone=True))
    last_active_at = Column(DateTime(timezone=True))
    
    notification_preferences = Column(JSON, default={})
    
    invited_by = Column(String, ForeignKey("users.id"))
    joined_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    removed_at = Column(DateTime(timezone=True))

    project = relationship("Project", back_populates="members")
    user = relationship("User", back_populates="project_memberships", foreign_keys=[user_id])
    # inviter = relationship("User", foreign_keys=[invited_by])
