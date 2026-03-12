from sqlalchemy import Column, String, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship
from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin

class GeneratedPDF(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "generated_pdfs"

    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    
    # Populated once the .pdf file has been written to disk
    file_path = Column(String(1000), nullable=True)
    file_url  = Column(String(1000), nullable=True)
    
    sections = relationship("PDFSection", back_populates="pdf", cascade="all, delete-orphan", order_by="PDFSection.order")

class PDFSection(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "pdf_sections"

    generated_pdf_id = Column(String, ForeignKey("generated_pdfs.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(500))
    content = Column(Text)
    section_type = Column(String(50), default="text") # text, list, table
    order = Column(Integer, default=0)
    
    pdf = relationship("GeneratedPDF", back_populates="sections")
