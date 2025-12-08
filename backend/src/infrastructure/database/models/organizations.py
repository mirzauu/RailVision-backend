from sqlalchemy import Column, String, Text, Integer, BigInteger, Boolean, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import relationship

from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin

# Use JSON for wider compatibility if needed, but schema asks for JSONB. 
# SQLAlchemy's JSON type usually maps to JSONB on Postgres and JSON on others.
# But expressly importing JSONB suggests Postgres optimization.
# Given database.py has sqlite fallback, I'll use JSON type standard which adapts.

class Organization(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "organizations"

    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    logo_url = Column(Text)
    website = Column(String(255))

    # Subscription & billing
    plan_type = Column(String(50), default='trial')
    subscription_status = Column(String(50), default='active', index=True)
    subscription_started_at = Column(DateTime(timezone=True))
    subscription_ends_at = Column(DateTime(timezone=True))
    billing_email = Column(String(255))

    # Usage limits
    max_users = Column(Integer, default=10)
    max_projects = Column(Integer, default=5)
    max_agents = Column(Integer, default=10)
    max_documents = Column(Integer, default=1000)
    max_storage_gb = Column(Integer, default=10)
    max_monthly_tokens = Column(BigInteger, default=1000000)

    # Settings & Metadata
    settings = Column(JSON, default={})
    metadata_ = Column("metadata", JSON, default={}) # 'metadata' is reserved in SQLAlchemy Base

    # Relationships
    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    agents = relationship("Agent", back_populates="organization", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="organization", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="organization", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="organization", cascade="all, delete-orphan")
    integrations = relationship("Integration", back_populates="organization", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="organization", cascade="all, delete-orphan")
    invitations = relationship("UserInvitation", back_populates="organization", cascade="all, delete-orphan")
