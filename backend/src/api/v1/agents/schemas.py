from pydantic import BaseModel
from typing import Dict, Optional, List
from datetime import datetime

class AgentCreate(BaseModel):
    name: str
    type: str
    config: Dict

class AgentResponse(BaseModel):
    id: str
    name: str
    type: str
    config: Optional[Dict]
    org_id: str
    created_at: datetime

    class Config:
        from_attributes = True
