from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from src.infrastructure.database.models.conversations import MessageRole, MessageStatus

class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    project_id: str
    org_id: str
    role: MessageRole
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    content: str
    status: MessageStatus
    attachments: List[str] | None = None
    created_at: datetime

    class Config:
        from_attributes = True

class ChatHistoryResponse(BaseModel):
    conversation_id: Optional[str] = None
    project_id: str
    messages: List[MessageResponse]
