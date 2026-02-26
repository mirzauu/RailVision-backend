from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_current_user
from src.config.database import get_db
from src.infrastructure.database.models import User
from src.infrastructure.database.models.conversations import Message
from src.infrastructure.database.models.knowledge_base import (
    KnowledgeBase,
    KnowledgeSourceType,
    KnowledgeStatus,
)
from src.api.v1.knowledge.schemas import (
    AddFromMessageRequest,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseResponse,
)

router = APIRouter()


# ── Helper ────────────────────────────────────────────────────────────────────

def _kb_to_response(kb: KnowledgeBase) -> KnowledgeBaseResponse:
    """Map ORM object → response schema (handles the metadata alias)."""
    return KnowledgeBaseResponse(
        id=kb.id,
        org_id=kb.org_id,
        title=kb.title,
        content=kb.content,
        summary=kb.summary,
        tags=kb.tags or [],
        category=kb.category,
        source_type=kb.source_type.value if kb.source_type else "manual",
        source_message_id=kb.source_message_id,
        source_conversation_id=kb.source_conversation_id,
        status=kb.status.value if kb.status else "active",
        vector_id=kb.vector_id,
        is_verified=kb.is_verified,
        priority=kb.priority,
        metadata=kb.metadata_ or {},
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


# ── POST /knowledge/from-message ──────────────────────────────────────────────

@router.post("/from-message", response_model=KnowledgeBaseResponse, status_code=201)
def add_from_message(
    body: AddFromMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a KnowledgeBase entry from an existing message.

    The endpoint looks up the message by `message_id`, validates that it
    belongs to the caller's organisation, then persists a KnowledgeBase row
    with the message content and a back-link to both the message and its
    parent conversation.

    Vector-DB indexing is intentionally **not** performed here – the entry
    is created with status ``active`` and can be indexed in a separate step.
    """
    org_id = current_user.org_id
    if not org_id:
        raise HTTPException(status_code=400, detail="User has no associated organization.")

    # Fetch the source message
    message: Optional[Message] = db.query(Message).filter(Message.id == body.message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail=f"Message '{body.message_id}' not found.")

    # Enforce org-level isolation
    if message.org_id != org_id:
        raise HTTPException(
            status_code=403,
            detail="Message does not belong to your organization.",
        )

    if not message.content or not message.content.strip():
        raise HTTPException(status_code=422, detail="Message has no content to add to knowledge base.")

    # ── Duplicate guard ────────────────────────────────────────────────────────
    existing = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.source_message_id == body.message_id,
            KnowledgeBase.org_id == org_id,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Message '{body.message_id}' has already been added to the knowledge base (entry id: {existing.id}).",
        )

    kb_entry = KnowledgeBase(
        org_id=org_id,
        title=body.title or _auto_title(message.content),
        content=message.content,
        summary=body.summary,
        tags=body.tags or [],
        category=body.category,
        source_type=KnowledgeSourceType.MESSAGE,
        source_message_id=message.id,
        source_conversation_id=message.conversation_id,
        status=KnowledgeStatus.ACTIVE,
        is_verified=False,
        priority=0,
        metadata_={},
    )

    db.add(kb_entry)
    db.commit()
    db.refresh(kb_entry)

    return _kb_to_response(kb_entry)


# ── POST /knowledge/ ──────────────────────────────────────────────────────────

@router.post("/", response_model=KnowledgeBaseResponse, status_code=201)
def create_knowledge_entry(
    body: KnowledgeBaseCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Manually create a knowledge-base entry (fact, note, etc.)."""
    org_id = current_user.org_id
    if not org_id:
        raise HTTPException(status_code=400, detail="User has no associated organization.")

    try:
        src_type = KnowledgeSourceType(body.source_type)
    except ValueError:
        src_type = KnowledgeSourceType.MANUAL

    kb_entry = KnowledgeBase(
        org_id=org_id,
        title=body.title or _auto_title(body.content),
        content=body.content,
        summary=body.summary,
        tags=body.tags or [],
        category=body.category,
        source_type=src_type,
        source_message_id=None,
        source_conversation_id=None,
        status=KnowledgeStatus.ACTIVE,
        is_verified=body.is_verified,
        priority=body.priority,
        metadata_=body.metadata_ or {},
    )

    db.add(kb_entry)
    db.commit()
    db.refresh(kb_entry)

    return _kb_to_response(kb_entry)


# ── GET /knowledge/ ───────────────────────────────────────────────────────────

@router.get("/", response_model=List[KnowledgeBaseResponse])
def list_knowledge_entries(
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all knowledge-base entries for the caller's organisation."""
    org_id = current_user.org_id
    if not org_id:
        return []

    q = db.query(KnowledgeBase).filter(KnowledgeBase.org_id == org_id)

    if category:
        q = q.filter(KnowledgeBase.category == category)
    if status:
        try:
            q = q.filter(KnowledgeBase.status == KnowledgeStatus(status))
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid status value: '{status}'")

    entries = (
        q.order_by(KnowledgeBase.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_kb_to_response(e) for e in entries]


# ── GET /knowledge/{knowledge_id} ─────────────────────────────────────────────

@router.get("/{knowledge_id}", response_model=KnowledgeBaseResponse)
def get_knowledge_entry(
    knowledge_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Fetch a single knowledge-base entry by ID."""
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge entry not found.")
    if kb.org_id != current_user.org_id:
        raise HTTPException(status_code=403, detail="Access denied.")
    return _kb_to_response(kb)


# ── DELETE /knowledge/{knowledge_id} ──────────────────────────────────────────

@router.delete("/{knowledge_id}", status_code=204)
def delete_knowledge_entry(
    knowledge_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a single knowledge-base entry."""
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge entry not found.")
    if kb.org_id != current_user.org_id:
        raise HTTPException(status_code=403, detail="Access denied.")
    db.delete(kb)
    db.commit()


# ── Internal helper ────────────────────────────────────────────────────────────

def _auto_title(content: str, max_len: int = 100) -> str:
    """Generate a title from the first line / sentence of content."""
    first_line = content.strip().split("\n")[0]
    return first_line[:max_len] + ("…" if len(first_line) > max_len else "")
