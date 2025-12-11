from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class OrgCreate(BaseModel):
    name: str

class OrgResponse(BaseModel):
    id: str
    name: str
    created_at: datetime

    class Config:
        from_attributes = True

class RoleLite(BaseModel):
    id: str
    name: str
    display_name: Optional[str] = None

    class Config:
        from_attributes = True

class OrgUserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    phone: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    status: str
    role: RoleLite
    created_at: datetime
    last_login_at: Optional[datetime] = None

    class Config:
        from_attributes = True
