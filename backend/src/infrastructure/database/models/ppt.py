from sqlalchemy import Column, String, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship
from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin

class Presentation(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "presentations"

    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(500), nullable=False)

    # Populated once the .pptx file has been written to disk
    file_path = Column(String(1000), nullable=True)
    file_url  = Column(String(1000), nullable=True)
    
    slides = relationship("PresentationSlide", back_populates="presentation", cascade="all, delete-orphan", order_by="PresentationSlide.order")

class PresentationSlide(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "presentation_slides"

    presentation_id = Column(String, ForeignKey("presentations.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(500))
    content = Column(Text)
    slide_type = Column(String(50), default="bullet") # bullet, text
    order = Column(Integer, default=0)
    
    presentation = relationship("Presentation", back_populates="slides")
