import enum
from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, DateTime, Text, ARRAY
from sqlalchemy.types import JSON, Enum as AlchemyEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin

class UserStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"

class InvitationStatus(str, enum.Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class Role(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "roles"

    name = Column(String(100), unique=True, nullable=False)
    display_name = Column(String(255), nullable=False)
    description = Column(Text)
    permissions = Column(JSON, nullable=False, default=[])
    is_system_role = Column(Boolean, default=False)
    is_default = Column(Boolean, default=False)

    users = relationship("User", back_populates="role")
    invitations = relationship("UserInvitation", back_populates="role")

class User(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "users"

    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    role_id = Column(String, ForeignKey("roles.id"), nullable=False)
    
    email = Column(String(255), nullable=False)
    email_verified = Column(Boolean, default=False)
    hashed_password = Column(String(255), nullable=False)
    
    full_name = Column(String(255))
    avatar_url = Column(Text)
    phone = Column(String(50))
    timezone = Column(String(50), default='UTC')
    language = Column(String(10), default='en')
    
    status = Column(AlchemyEnum(UserStatus, name="user_status"), default=UserStatus.ACTIVE, nullable=False, index=True)
    
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(Text)
    backup_codes = Column(JSON, default=[])
    
    last_login_at = Column(DateTime(timezone=True))
    last_active_at = Column(DateTime(timezone=True))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))
    
    preferences = Column(JSON, default={})
    metadata_ = Column("metadata", JSON, default={})

    # Relationships
    organization = relationship("Organization", back_populates="users")
    role = relationship("Role", back_populates="users")
    
    projects_created = relationship("Project", back_populates="creator", foreign_keys="[Project.created_by]")
    project_memberships = relationship(
        "ProjectMember",
        back_populates="user",
        cascade="all, delete-orphan",
        primaryjoin="User.id == ProjectMember.user_id",
        foreign_keys="[ProjectMember.user_id]",
    )
    messages = relationship("Message", back_populates="user")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship(
        "Notification",
        back_populates="user",
        cascade="all, delete-orphan",
        primaryjoin="User.id == Notification.user_id",
        foreign_keys="[Notification.user_id]",
    )
    # invitations_sent = relationship("UserInvitation", back_populates="inviter", foreign_keys="[UserInvitation.invited_by]")
    
    # We delay defining some relationships that create circular imports issues or complexity until the other models are defined
    # But string based works. `Project` model will be in `projects.py`.

class UserInvitation(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "user_invitations"

    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    role_id = Column(String, ForeignKey("roles.id"), nullable=False)
    invited_by = Column(String, ForeignKey("users.id"), nullable=False)
    
    project_ids = Column(JSON, default=[])
    token = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    
    status = Column(AlchemyEnum(InvitationStatus, name="invitation_status"), default=InvitationStatus.PENDING, nullable=False)
    accepted_at = Column(DateTime(timezone=True))
    accepted_by = Column(String, ForeignKey("users.id"))
    invitation_message = Column(Text)

    organization = relationship("Organization", back_populates="invitations")
    role = relationship("Role", back_populates="invitations")
    inviter = relationship("User", foreign_keys=[invited_by])
    accepter = relationship("User", foreign_keys=[accepted_by])

class RefreshToken(Base, UUIDMixin):
    __tablename__ = "refresh_tokens"
    # Mixins only provide created_at, updated_at, deleted_at. Refresh tokens only need created_at and specific fields.
    # Actually DB design says created_at.

    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(255), unique=True, nullable=False, index=True)
    
    device_name = Column(String(255))
    ip_address = Column(String) # INET in postgres, string here for generic
    user_agent = Column(Text)
    
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked_at = Column(DateTime(timezone=True))
    revoked_reason = Column(String(100))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="refresh_tokens")
