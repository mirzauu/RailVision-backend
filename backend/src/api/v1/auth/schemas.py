from pydantic import BaseModel, EmailStr
from typing import Optional

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    org_id: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class UserDetail(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    org_id: str
    role_id: str
    status: str
    avatar_url: Optional[str] = None

    class Config:
        from_attributes = True

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserDetail

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    role_id: str
    status: str

    class Config:
        from_attributes = True
