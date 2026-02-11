from pydantic import BaseModel
from typing import List, Optional, Any

class ChatHistoryResponse(BaseModel):
    conversation_id: Optional[str] = None
    project_id: str
    messages: List[Any]
    presentations: List[Any]
    generated_pdfs: List[Any]
    generated_word_docs: List[Any]
    
    class Config:
        from_attributes = True
