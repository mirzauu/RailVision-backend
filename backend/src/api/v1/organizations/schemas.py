from pydantic import BaseModel
from datetime import datetime

class OrgCreate(BaseModel):
    name: str

class OrgResponse(BaseModel):
    id: str
    name: str
    created_at: datetime

    class Config:
        from_attributes = True
