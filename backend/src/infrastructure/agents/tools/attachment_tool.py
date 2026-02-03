from typing import List, Optional, Any
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from src.application.attachments.service import AttachmentService
from src.infrastructure.database.models.conversations import Message, Conversation
from src.infrastructure.database.models.documents import Document, DocumentScope

class AttachmentToolInput(BaseModel):
    query: str = Field(description="The question or query to search for in the conversation's attached documents.")

class AttachmentTool:
    """Tool for retrieving information from files attached to the current conversation."""

    name = "search_attachments"
    description = """
    Use this tool to search for specific information within documents that the user has attached to THIS conversation.
    This is useful for answering questions based on the content of uploaded PDFs, text files, or other supported documents 
    provided by the user in the chat.
    """

    def __init__(self, sql_db: Session, user_id: str, conversation_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.attachment_service = AttachmentService()

    async def arun(self, query: str) -> str:
        """Retrieve information from conversation attachments."""
        try:
            # 1. Find all messages in the conversation to collect attachment IDs
            messages = self.sql_db.query(Message).filter(
                Message.conversation_id == self.conversation_id
            ).all()

            attachment_ids = []
            for msg in messages:
                if msg.attachments:
                    for att in msg.attachments:
                        if isinstance(att, dict) and "id" in att:
                            attachment_ids.append(att["id"])
                        elif isinstance(att, str):
                            attachment_ids.append(att)
            
            # 2. Find project_id associated with this conversation
            conv = self.sql_db.query(Conversation).filter(Conversation.id == self.conversation_id).first()
            if conv and conv.project_id:
                # Find all documents linked to this project
                project_docs = self.sql_db.query(Document).filter(
                    Document.project_id == conv.project_id
                ).all()
                for doc in project_docs:
                    attachment_ids.append(doc.id)
            
            # Remove duplicates
            attachment_ids = list(set(attachment_ids))

            if not attachment_ids:
                return "No attachments found in this conversation to search."

            # 2. Retrieve context from these attachments
            context = self.attachment_service.retrieve_context_for_attachments(
                query=query,
                attachment_ids=attachment_ids,
                top_k=5
            )

            if not context:
                return "No relevant information found in the attached documents for the given query."

            return f"Context found in attachments:\n\n{context}"

        except Exception as e:
            return f"Error searching attachments: {str(e)}"

    def run(self, query: str) -> str:
        """Synchronous wrapper (not ideally used but required by some frameworks)."""
        return "Error: Please use the async version (arun) of this tool."

def attachment_search_tool(sql_db: Session, user_id: str, conversation_id: Optional[str]) -> Optional[StructuredTool]:
    """Create and return the attachment search tool."""
    if not conversation_id:
        return None
        
    tool_instance = AttachmentTool(sql_db, user_id, conversation_id)
    return StructuredTool.from_function(
        func=tool_instance.run,
        coroutine=tool_instance.arun,
        name=tool_instance.name,
        description=tool_instance.description,
        args_schema=AttachmentToolInput,
    )
