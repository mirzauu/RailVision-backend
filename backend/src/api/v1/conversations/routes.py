from fastapi import APIRouter, Depends, Body, HTTPException
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
        os.environ["CHAT_MODEL"] = body.model

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
    body: ChatRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.model and "/" in body.model and body.model.strip().lower() not in {"string", "null", "none"}:
        os.environ["CHAT_MODEL"] = body.model

    if body.framework == "cso" and body.agent and body.agent != "auto":
        provider = ProviderService.create(user_id=str(current_user.id))
        tools = ToolService(db, str(current_user.id))
        
        # Use RouterAgent to get the specific agent directly
        router_agent = CSORouterAgent(provider, tools)
        # Check if the agent exists
        if not router_agent.get_agent(body.agent):
            return {"error": f"Agent {body.agent} not found"}

        target_agent = router_agent.get_agent(body.agent)

        # Build conversation and history (persist user message)
        project = db.query(Project).filter(Project.id == body.project_id).first() if body.project_id else None
        if project:
            conv = db.query(Conversation).filter(Conversation.project_id == project.id).first()
            if not conv:
                conv = Conversation(project_id=project.id, org_id=current_user.org_id, title="Conversation")
                db.add(conv)
                db.commit()
                db.refresh(conv)
        else:
            conv = db.query(Conversation).filter(Conversation.org_id == current_user.org_id).first()
            if not conv:
                raise HTTPException(status_code=400, detail="conversation requires a valid project")

        user_msg = Message(
            conversation_id=conv.id,
            project_id=conv.project_id,
            org_id=current_user.org_id,
            role=MessageRole.USER,
            user_id=str(current_user.id),
            content=body.query,
            status=MessageStatus.SENT,
            attachments=[body.attachment] if body.attachment else [],
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
        history = [m.content for m in msgs if m.content]

        ctx = ChatContext(
            history=history,
            query=body.query,
            additional_context=body.attachment or ""
        )

        async def stream_cso_agent():
            full: List[str] = []
            async for chunk in target_agent.run_stream(ctx):
                if chunk.response:
                    full.append(chunk.response)
                yield json.dumps(chunk.model_dump(), default=str) + "\n"
            ai_msg = Message(
                conversation_id=conv.id,
                project_id=conv.project_id,
                org_id=current_user.org_id,
                role=MessageRole.ASSISTANT,
                agent_id=None,
                content="".join(full),
                status=MessageStatus.SENT,
            )
            db.add(ai_msg)
            db.commit()
            db.refresh(ai_msg)

        return StreamingResponse(stream_cso_agent(), media_type="application/json")

    service = ConversationService(ProviderService.create(user_id=str(current_user.id)))

    async def stream_response():
        async for chunk in service.chat_stream(
            db=db,
            user_id=str(current_user.id),
            org_id=current_user.org_id,
            query=body.query,
            project_id=body.project_id,
            framework=body.framework,
            model=body.model,
            agent=body.agent,
            attachment=body.attachment,
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
