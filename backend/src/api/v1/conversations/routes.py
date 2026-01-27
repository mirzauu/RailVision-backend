from fastapi import APIRouter, Depends, Body, HTTPException, File, UploadFile, Form
import os
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import List, Optional
from src.config.database import get_db
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User
from src.application.conversations.service import ConversationService
from src.infrastructure.llm.provider_service import ProviderService
from src.api.v1.conversations.schemas import ChatHistoryResponse
from src.application.agents.cso.router_agent import CSORouterAgent
from src.application.tools.service import ToolService
from src.domain.agents.base import ChatContext
from src.infrastructure.database.models.projects import Project
from src.infrastructure.database.models.conversations import Conversation, Message, MessageRole, MessageStatus
from src.application.attachments.service import AttachmentService

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    project_id: str = "default"
    framework: str = "pydantic"
    model: str | None = None
    agent: str | None = None
    attachment: str | None = None


@router.post("/chat")
async def chat(
    body: ChatRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.model and "/" in body.model and body.model.strip().lower() not in {"string", "null", "none"}:
        os.environ["CHAT_MODEL"] = "anthropic/claude-3-haiku-20240307"

    service = ConversationService(ProviderService.create(user_id=str(current_user.id)))
    resp = await service.chat(
        db=db,
        user_id=str(current_user.id),
        org_id=current_user.org_id,
        query=body.query,
        project_id=body.project_id,
        framework=body.framework,
        model=body.model,
        agent=body.agent,
        attachment=body.attachment,
    )
    return {"response": resp.response}


@router.post("/chat/stream")
async def chat_stream(
    query: str = Form(...),
    project_id: str = Form("default"),
    framework: str = Form("pydantic"),
    model: Optional[str] = Form(None),
    agent: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Chat endpoint with streaming response.
    
    Supports optional file attachments. When a file is provided:
    1. The file is parsed and indexed to Pinecone
    2. Relevant context is retrieved using RAG
    3. Context is passed to the AI agent
    """
    if model and "/" in model and model.strip().lower() not in {"string", "null", "none"}:
        os.environ["CHAT_MODEL"] = model

    user_id = str(current_user.id)
    org_id = current_user.org_id
    
    # Process file attachment if provided
    attachment_context = ""
    attachment_id = None
    if file and file.filename:
        attachment_service = AttachmentService()
        file_bytes = await file.read()
        
        # Process and index the attachment
        attachment_id = await attachment_service.process_attachment(
            db=db,
            file_bytes=file_bytes,
            filename=file.filename,
            user_id=user_id,
            org_id=org_id,
            project_id=project_id if project_id != "default" else None
        )
        
        # Retrieve relevant context from the attachment
        attachment_context = attachment_service.retrieve_attachment_context(
            query=query,
            attachment_id=attachment_id,
            top_k=5,
        )

    if framework == "cso" and agent and agent != "auto":
        provider = ProviderService.create(user_id=user_id)
        tools = ToolService(db, user_id)
        
        # Use RouterAgent to get the specific agent directly
        router_agent = CSORouterAgent(provider, tools)
        # Check if the agent exists
        if not router_agent.get_agent(agent):
            return {"error": f"Agent {agent} not found"}

        target_agent = router_agent.get_agent(agent)

        # Build conversation and history (persist user message)
        project = db.query(Project).filter(Project.id == project_id).first() if project_id else None
        if project:
            conv = db.query(Conversation).filter(Conversation.project_id == project.id).first()
            if not conv:
                conv = Conversation(project_id=project.id, org_id=org_id, title="Conversation")
                db.add(conv)
                db.commit()
                db.refresh(conv)
        else:
            conv = db.query(Conversation).filter(Conversation.org_id == org_id).first()
            if not conv:
                raise HTTPException(status_code=400, detail="conversation requires a valid project")

        user_msg = Message(
            conversation_id=conv.id,
            project_id=conv.project_id,
            org_id=org_id,
            role=MessageRole.USER,
            user_id=user_id,
            content=query,
            status=MessageStatus.SENT,
            attachments=[attachment_id] if attachment_id else [],
        )
        db.add(user_msg)
        db.commit()
        db.refresh(user_msg)

        msgs = (
            db.query(Message)
            .filter(Message.project_id == conv.project_id)
            .order_by(Message.created_at.asc())
            .limit(20)
            .all()
        )
        history = []
        # Exclude the message we just added to avoid duplication in history
        history_msgs = msgs[:-1] if msgs else []
        for m in history_msgs:
            if m.content:
                role = "user" if m.role == MessageRole.USER else "assistant"
                history.append({"role": role, "content": m.content})

        ctx = ChatContext(
            history=history,
            query=query,
            additional_context=attachment_context
        )

        # Capture IDs for the generator to avoid DetachedInstanceError
        conv_id = conv.id
        conv_project_id = conv.project_id

        async def stream_cso_agent():
            full: List[str] = []
            async for chunk in target_agent.run_stream(ctx):
                if chunk.response:
                    full.append(chunk.response)
                yield json.dumps(chunk.model_dump(), default=str) + "\n"
            
            # Re-fetch or use captured IDs to avoid detachment
            ai_msg = Message(
                conversation_id=conv_id,
                project_id=conv_project_id,
                org_id=org_id,
                role=MessageRole.ASSISTANT,
                agent_id=None,
                content="".join(full),
                status=MessageStatus.SENT,
            )
            db.add(ai_msg)
            db.commit()
            db.refresh(ai_msg)

        return StreamingResponse(stream_cso_agent(), media_type="application/json")

    service = ConversationService(ProviderService.create(user_id=user_id))

    async def stream_response():
        async for chunk in service.chat_stream(
            db=db,
            user_id=user_id,
            org_id=org_id,
            query=query,
            project_id=project_id,
            framework=framework,
            model=model,
            agent=agent,
            attachment=attachment_context,
            attachment_id=attachment_id,
        ):
            yield json.dumps(chunk.model_dump(), default=str) + "\n"

    return StreamingResponse(stream_response(), media_type="application/json")


@router.get("/history/{project_id}", response_model=ChatHistoryResponse)
def get_history(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ConversationService(ProviderService.create(user_id=str(current_user.id)))
    return service.get_chat_history(db=db, org_id=current_user.org_id, project_id=project_id)
