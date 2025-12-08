import uuid
from sqlalchemy import Column, DateTime, String
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_mixin

@declarative_mixin
class TimestampMixin:
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

@declarative_mixin
class UUIDMixin:
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
