from fastapi import APIRouter, Depends, Body
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
            yield json.dumps(chunk.model_dump()) + "\n"

    return StreamingResponse(stream_response(), media_type="application/json")

@router.get("/history/{project_id}", response_model=ChatHistoryResponse)
def get_history(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ConversationService(ProviderService.create(user_id=str(current_user.id)))
    return service.get_chat_history(db=db, org_id=current_user.org_id, project_id=project_id)
