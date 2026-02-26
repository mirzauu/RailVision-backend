import enum
from sqlalchemy import Column, String, Text, Boolean, ForeignKey, Integer
from sqlalchemy.types import JSON, Enum as AlchemyEnum
from sqlalchemy.orm import relationship

from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin


class KnowledgeSourceType(str, enum.Enum):
    MESSAGE = "message"
    MANUAL = "manual"
    DOCUMENT = "document"
    EXTERNAL = "external"


class KnowledgeStatus(str, enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    PENDING_INDEXING = "pending_indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class KnowledgeBase(Base, UUIDMixin, TimestampMixin):
    """
    Stores organizational knowledge/facts extracted from messages,
    conversations, or added manually. Each entry represents a discrete
    fact or piece of knowledge that can later be indexed into a vector DB.
    """
    __tablename__ = "knowledge_base"

    # ── Ownership ──────────────────────────────────────────────────────────────
    org_id = Column(
        String,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Content ────────────────────────────────────────────────────────────────
    title = Column(String(500), nullable=True)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)         # short human-readable summary
    tags = Column(JSON, default=[])
    category = Column(String(200), nullable=True)  # e.g. "pricing", "logistics"

    # ── Source tracing ─────────────────────────────────────────────────────────
    source_type = Column(
        AlchemyEnum(KnowledgeSourceType, name="knowledge_source_type"),
        default=KnowledgeSourceType.MANUAL,
        nullable=False,
        index=True,
    )

    # FK to the message that originated this knowledge entry (nullable)
    source_message_id = Column(
        String,
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # FK to the conversation the message belongs to (nullable, for convenience)
    source_conversation_id = Column(
        String,
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ── Status & vector-DB readiness ───────────────────────────────────────────
    status = Column(
        AlchemyEnum(KnowledgeStatus, name="knowledge_status"),
        default=KnowledgeStatus.ACTIVE,
        nullable=False,
        index=True,
    )

    # Will be populated after vector-DB indexing
    vector_id = Column(String(500), nullable=True)

    is_verified = Column(Boolean, default=False)   # human-reviewed & confirmed
    priority = Column(Integer, default=0)           # higher = more important

    # Extra arbitrary metadata (e.g. confidence score, agent id, etc.)
    metadata_ = Column("metadata", JSON, default={})

    # ── Relationships ──────────────────────────────────────────────────────────
    organization = relationship("Organization", back_populates="knowledge_entries")
    source_message = relationship("Message", foreign_keys=[source_message_id])
    source_conversation = relationship(
        "Conversation", foreign_keys=[source_conversation_id]
    )
