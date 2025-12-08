import enum
from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, Text, ARRAY, BigInteger, DateTime, Numeric
from sqlalchemy.types import JSON, Enum as AlchemyEnum
from sqlalchemy.orm import relationship

from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin

class DocumentType(str, enum.Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    CSV = "csv"
    TXT = "txt"
    MD = "md"
    JSON = "json"
    XML = "xml"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    URL = "url"
    EMAIL = "email"
    SLACK_THREAD = "slack_thread"
    CODE = "code"

class DocumentStatus(str, enum.Enum):
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INGESTED = "ingested"
    FAILED = "failed"
    ARCHIVED = "archived"

class DocumentScope(str, enum.Enum):
    ORGANIZATION = "organization"
    PROJECT = "project"
    AGENT = "agent"
    PRIVATE = "private"

class Document(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "documents"

    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    
    uploaded_by = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_type = Column(AlchemyEnum(DocumentType, name="document_type"), nullable=False)
    mime_type = Column(String(255))
    file_size_bytes = Column(BigInteger, nullable=False)
    checksum = Column(String(64), index=True)
    
    storage_path = Column(Text, nullable=False)
    storage_backend = Column(String(50), default='s3')
    storage_url = Column(Text)
    
    status = Column(AlchemyEnum(DocumentStatus, name="document_status"), default=DocumentStatus.UPLOADED, nullable=False, index=True)
    ingestion_started_at = Column(DateTime(timezone=True))
    ingestion_completed_at = Column(DateTime(timezone=True))
    ingestion_error = Column(Text)
    retry_count = Column(Integer, default=0)
    
    extracted_text = Column(Text)
    text_length = Column(Integer)
    language = Column(String(10))
    page_count = Column(Integer)
    
    embedding_model = Column(String(100))
    chunks_count = Column(Integer, default=0)
    embeddings_generated_at = Column(DateTime(timezone=True))
    
    title = Column(String(500))
    description = Column(Text)
    author = Column(String(255))
    source_url = Column(Text)
    source_integration = Column(String(50))
    external_id = Column(String(255))
    
    tags = Column(JSON, default=[])
    category = Column(String(100))
    keywords = Column(JSON, default=[])
    
    scope = Column(AlchemyEnum(DocumentScope, name="document_scope"), default=DocumentScope.ORGANIZATION, nullable=False, index=True)
    assigned_agent_ids = Column(JSON, default=[])
    shared_with_project_ids = Column(JSON, default=[])
    
    version = Column(Integer, default=1)
    parent_document_id = Column(String, ForeignKey("documents.id"))
    is_latest_version = Column(Boolean, default=True)
    
    view_count = Column(Integer, default=0)
    download_count = Column(Integer, default=0)
    citation_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime(timezone=True))
    
    expires_at = Column(DateTime(timezone=True))
    
    metadata_ = Column("metadata", JSON, default={})

    # Relationships
    organization = relationship("Organization", back_populates="documents")
    project = relationship("Project", back_populates="documents")
    uploader = relationship("User", foreign_keys=[uploaded_by])
    
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    parent_document = relationship("Document", remote_side="[Document.id]", backref="versions")

class DocumentChunk(Base, UUIDMixin):
    __tablename__ = "document_chunks"
    
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(String, ForeignKey("projects.id"), index=True)
    
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_tokens = Column(Integer, nullable=False)
    
    pinecone_id = Column(String(255), unique=True, index=True)
    embedding_model = Column(String(100))
    embedding_dimension = Column(Integer, default=1536)
    
    preceding_text = Column(Text)
    following_text = Column(Text)
    
    page_number = Column(Integer)
    section_title = Column(String(500))
    paragraph_index = Column(Integer)
    
    chunk_metadata = Column(JSON, default={})
    
    access_count = Column(Integer, default=0)
    citation_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime(timezone=True))
    avg_relevance_score = Column(Numeric(5, 4))
    
    from sqlalchemy.sql import func
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    document = relationship("Document", back_populates="chunks")
