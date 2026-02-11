from sqlalchemy import Column, String, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship
from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin

class GeneratedWord(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "generated_word_docs"

    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    
    sections = relationship("WordSection", back_populates="word_doc", cascade="all, delete-orphan", order_by="WordSection.order")

class WordSection(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "word_sections"

    generated_word_id = Column(String, ForeignKey("generated_word_docs.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(500))
    content = Column(Text)
    section_type = Column(String(50), default="text") # text, list, table
    order = Column(Integer, default=0)
    
    word_doc = relationship("GeneratedWord", back_populates="sections")
