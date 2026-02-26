from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Shared ────────────────────────────────────────────────────────────────────

class KnowledgeBaseBase(BaseModel):
    title: Optional[str] = None
    content: str
    summary: Optional[str] = None
    tags: List[str] = []
    category: Optional[str] = None
    is_verified: bool = False
    priority: int = 0
    metadata_: Dict[str, Any] = Field(default_factory=dict, alias="metadata")

    class Config:
        populate_by_name = True


# ── Request: add from a message ───────────────────────────────────────────────

class AddFromMessageRequest(BaseModel):
    """
    Pass a message_id; the API fetches its content and creates a
    KnowledgeBase entry linked to that message and its conversation.
    """
    message_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = []
    category: Optional[str] = None


# ── Request: create manually ──────────────────────────────────────────────────

class KnowledgeBaseCreateRequest(KnowledgeBaseBase):
    source_type: str = "manual"


# ── Response ──────────────────────────────────────────────────────────────────

class KnowledgeBaseResponse(BaseModel):
    id: str
    org_id: str
    title: Optional[str]
    content: str
    summary: Optional[str]
    tags: List[str]
    category: Optional[str]
    source_type: str
    source_message_id: Optional[str]
    source_conversation_id: Optional[str]
    status: str
    vector_id: Optional[str]
    is_verified: bool
    priority: int
    metadata_: Dict[str, Any] = Field(default_factory=dict, alias="metadata")
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True
