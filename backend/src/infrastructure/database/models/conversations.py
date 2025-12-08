import enum
from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, Text, ARRAY, Numeric, BigInteger, DateTime
from sqlalchemy.types import JSON, Enum as AlchemyEnum
from sqlalchemy.orm import relationship

from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin

class ConversationType(str, enum.Enum):
    LINEAR = "linear"
    THREADED = "threaded"
    WORKFLOW = "workflow"

class ConversationStatus(str, enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    HUMAN_FEEDBACK = "human_feedback"

class MessageStatus(str, enum.Enum):
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    ERROR = "error"
    DELETED = "deleted"

class Conversation(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "conversations"

    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    
    title = Column(String(500))
    description = Column(Text)
    type = Column(AlchemyEnum(ConversationType, name="conversation_type"), default=ConversationType.LINEAR, nullable=False)
    status = Column(AlchemyEnum(ConversationStatus, name="conversation_status"), default=ConversationStatus.ACTIVE, nullable=False, index=True)
    
    parent_conversation_id = Column(String, ForeignKey("conversations.id"), index=True)
    thread_depth = Column(Integer, default=0)
    
    started_by_user_id = Column(String, ForeignKey("users.id"))
    started_by_agent_id = Column(String, ForeignKey("agents.id"))
    
    primary_agent_id = Column(String, ForeignKey("agents.id"), index=True)
    participating_agent_ids = Column(JSON, default=[])
    
    # Tracking
    message_count = Column(Integer, default=0)
    total_tokens_used = Column(BigInteger, default=0)
    total_cost_usd = Column(Numeric(10, 4), default=0)
    
    summary = Column(Text)
    summary_generated_at = Column(DateTime(timezone=True))
    key_points = Column(JSON, default=[])
    
    tags = Column(JSON, default=[])
    
    is_pinned = Column(Boolean, default=False)
    priority = Column(String(50), default='normal')
    
    last_message_at = Column(DateTime(timezone=True), index=True)
    last_message_by_user_id = Column(String, ForeignKey("users.id"))
    last_message_by_agent_id = Column(String, ForeignKey("agents.id"))
    
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime(timezone=True))
    resolved_by = Column(String, ForeignKey("users.id"))
    resolution_notes = Column(Text)
    
    metadata_ = Column("metadata", JSON, default={})

    # Relationships
    project = relationship("Project", back_populates="conversations")
    organization = relationship("Organization", back_populates="conversations")
    
    # initiators
    started_by_user = relationship("User", foreign_keys=[started_by_user_id])
    started_by_agent = relationship("Agent", foreign_keys=[started_by_agent_id])
    
    primary_agent = relationship("Agent", foreign_keys=[primary_agent_id])
    
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    parent_conversation = relationship(
        "Conversation",
        remote_side="[Conversation.id]",
        backref="children",
    )

class Message(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "messages"

    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    
    role = Column(AlchemyEnum(MessageRole, name="message_role"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    agent_id = Column(String, ForeignKey("agents.id"), index=True)
    
    content = Column(Text, nullable=False)
    content_type = Column(String(50), default='text')
    
    attachments = Column(JSON, default=[])
    
    reply_to_message_id = Column(String, ForeignKey("messages.id"), index=True)
    
    status = Column(AlchemyEnum(MessageStatus, name="message_status"), default=MessageStatus.SENT, nullable=False)
    error_message = Column(Text)
    error_code = Column(String(100))
    
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    cost_usd = Column(Numeric(10, 6))
    
    llm_provider = Column(String(50))
    llm_model = Column(String(100))
    llm_temperature = Column(Numeric(3, 2))
    response_time_ms = Column(Integer)
    time_to_first_token_ms = Column(Integer)
    
    context_used = Column(JSON, default={})
    
    user_rating = Column(Integer)
    user_feedback = Column(Text)
    feedback_at = Column(DateTime(timezone=True))
    is_helpful = Column(Boolean)
    
    mentions_user_ids = Column(JSON, default=[])
    mentions_agent_ids = Column(JSON, default=[])
    
    read_by_user_ids = Column(JSON, default=[])
    
    metadata_ = Column("metadata", JSON, default={})
    
    edited_at = Column(DateTime(timezone=True))
    edit_history = Column(JSON, default=[])

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    project = relationship("Project") # Avoiding circular back_populates for now or just one-way
    # organization = relationship("Organization") # Not necessary to populate usually
    
    user = relationship("User", back_populates="messages")
    agent = relationship("Agent", back_populates="messages")
    
    reply_to = relationship("Message", remote_side="[Message.id]", backref="replies")
