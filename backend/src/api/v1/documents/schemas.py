from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any
from datetime import datetime

class DocumentResponse(BaseModel):
    id: str
    org_id: str
    project_id: Optional[str] = None
    uploaded_by: str = Field(validation_alias="uploader")
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

    @field_validator('uploaded_by', mode='before')
    @classmethod
    def get_uploader_name(cls, v: Any) -> str:
        if hasattr(v, 'full_name'):
            return v.full_name or v.email or "Unknown"
        if hasattr(v, 'email'):
            return v.email
        return str(v) if v else "Unknown"

    class Config:
        from_attributes = True
