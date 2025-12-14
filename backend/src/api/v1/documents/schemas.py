from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class DocumentResponse(BaseModel):
    id: str
    org_id: str
    project_id: Optional[str] = None
    uploaded_by: str
    filename: str
    original_filename: str
    file_type: str
    mime_type: Optional[str] = None
    file_size_bytes: int
    storage_path: str
    storage_backend: str
    status: str
    title: Optional[str] = None
    description: Optional[str] = None
    scope: str
    assigned_agent_ids: List[str]
    category: Optional[str] = None
    tags: List[str]
    created_at: datetime

    class Config:
        from_attributes = True
