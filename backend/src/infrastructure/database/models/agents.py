import enum
from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, Text, ARRAY, Numeric
from sqlalchemy.types import JSON, Enum as AlchemyEnum
from sqlalchemy.orm import relationship

from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin

class AgentType(str, enum.Enum):
    CSO = "cso"
    CRO = "cro"
    CFO = "cfo"
    COO = "coo"
    CHRO = "chro"
    CTO = "cto"
    CMO = "cmo"
    CLO = "clo"
    GENERAL = "general"
    CUSTOM = "custom"

class AgentStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    DEPRECATED = "deprecated"

class Agent(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "agents"

    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    
    type = Column(AlchemyEnum(AgentType, name="agent_type"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    description = Column(Text)
    avatar_url = Column(Text)
    
    parent_agent_id = Column(String, ForeignKey("agents.id", ondelete="SET NULL"), index=True)
    
    config = Column(JSON, nullable=False, default={})
    
    status = Column(AlchemyEnum(AgentStatus, name="agent_status"), default=AgentStatus.ACTIVE, nullable=False, index=True)
    is_visible = Column(Boolean, default=True)
    
    can_collaborate = Column(Boolean, default=True)
    can_delegate = Column(Boolean, default=True)
    
    specialization = Column(String(255))
    expertise_areas = Column(JSON, default=[])
    
    # Performance tracking
    total_conversations = Column(Integer, default=0)
    total_messages = Column(Integer, default=0)
    avg_response_time_ms = Column(Integer)
    avg_satisfaction_rating = Column(Numeric(3, 2))
    
    version = Column(Integer, default=1)
    metadata_ = Column("metadata", JSON, default={})

    # Relationships
    organization = relationship("Organization", back_populates="agents")
    parent_agent = relationship("Agent", remote_side="[Agent.id]", backref="sub_agents")
    
    project_assignments = relationship("ProjectAgent", back_populates="agent", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="agent")
    
    # We will need relationships for logging later (collaboration logs, usage logs)
