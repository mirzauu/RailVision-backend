from fastapi import APIRouter, Depends, Body, HTTPException, File, UploadFile, Form
import os
from fastapi.responses import StreamingResponse, JSONResponse
import json
from datetime import datetime
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import List, Optional, Any
from src.infrastructure.agents.reasoning_manager import reset_reasoning_manager, finalize_reasoning
from src.config.database import get_db
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User
from src.infrastructure.database.models.documents import Document
from src.application.conversations.service import ConversationService
from src.infrastructure.llm.provider_service import ProviderService
from src.api.v1.conversations.schemas import ChatHistoryResponse
from src.application.agents.cso.router_agent import CSORouterAgent
from src.application.agents.cco.router_agent import CCORouterAgent
from src.application.agents.cfo.router_agent import CFORouterAgent
from src.application.agents.coo.router_agent import COORouterAgent
from src.application.agents.chro.router_agent import CHRORouterAgent
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
    body.query = body.query.replace("\x00", "")
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
    file: List[UploadFile] = File(default=[]),
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
    
    # Sanitize query to remove NULL bytes which PostgreSQL doesn't support
    query = query.replace("\x00", "")
    
    # Auto-create project if none provided
    if not project_id or project_id == "default":
        from src.api.v1.projects.routes import generate_project_name
        from src.application.projects.service import ProjectService
        from src.infrastructure.database.repositories.project_repository import ProjectRepository
        from src.infrastructure.database.repositories.project_agent_repository import ProjectAgentRepository
        from src.infrastructure.database.repositories.project_member_repository import ProjectMemberRepository
        
        project_name = await generate_project_name(query, user_id)
        
        project_svc = ProjectService(
            ProjectRepository(db),
            ProjectAgentRepository(db),
            ProjectMemberRepository(db),
        )
        
        new_project = project_svc.create_project(
            org_id=org_id,
            created_by=user_id,
            name=project_name,
            description=query[:500],
            agent_id=agent if agent and agent != "auto" else None
        )
        project_id = str(new_project.id)

    # Process file attachment if provided
    attachment_context = ""
    attachment_ids = []
    attachment_infos = []

    if file:
        attachment_service = AttachmentService()
        for idx, f in enumerate(file, 1):
            if not f.filename:
                continue
                
            file_bytes = await f.read()
            
            # Process and index the attachment
            att_id = await attachment_service.process_attachment(
                db=db,
                file_bytes=file_bytes,
                filename=f.filename,
                user_id=user_id,
                org_id=org_id,
                project_id=project_id if project_id != "default" else None
            )
            
            if att_id:
                attachment_ids.append(att_id)
                # Retrieve relevant context from the attachment
                context = attachment_service.retrieve_attachment_context(
                    query=query,
                    attachment_id=att_id,
                    top_k=5,
                )
                attachment_context += f"\nuser attached file {idx}- {f.filename} : {context}"
                
                # Get attachment info to include in response
                attachment_doc = db.query(Document).filter(Document.id == att_id).first()
                if attachment_doc:
                    attachment_infos.append({
                        "id": attachment_doc.id,
                        "filename": attachment_doc.original_filename,
                        "file_type": str(attachment_doc.file_type),
                        "file_size_bytes": attachment_doc.file_size_bytes,
                        "status": str(attachment_doc.status)
                    })
        
        if attachment_context:
            attachment_context = f"user attached docs with the user query, the attachment content is: {attachment_context}"

    if framework in ("cso", "cco", "cfo", "coo", "chro") and agent and agent != "auto":
        provider = ProviderService.create(user_id=user_id)
        
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

        conv_id = conv.id
        conv_project_id = conv.project_id
        
        tools = ToolService(db, user_id, conversation_id=conv_id)
        
        # Use the appropriate RouterAgent based on the framework
        if framework == "cco":
            router_agent = CCORouterAgent(provider, tools)
        elif framework == "cfo":
            router_agent = CFORouterAgent(provider, tools)
        elif framework == "coo":
            router_agent = COORouterAgent(provider, tools)
        elif framework == "chro":
            router_agent = CHRORouterAgent(provider, tools)
        else:
            router_agent = CSORouterAgent(provider, tools)
        # Check if the agent exists
        if not router_agent.get_agent(agent):
            return {"error": f"Agent {agent} not found"}

        target_agent = router_agent.get_agent(agent)

        user_msg = Message(
            conversation_id=conv_id,
            project_id=conv_project_id,
            org_id=org_id,
            role=MessageRole.USER,
            user_id=user_id,
            content=query,
            status=MessageStatus.SENT,
            attachments=attachment_infos,
        )
        db.add(user_msg)
        db.commit()
        db.refresh(user_msg)

        msgs = (
            db.query(Message)
            .filter(Message.project_id == conv_project_id)
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

        async def stream_cso_agent():
            full: List[str] = []
            first_chunk = True
            reset_reasoning_manager()
            try:
                async for chunk in target_agent.run_stream(ctx):
                    if chunk.response:
                        full.append(chunk.response)
                    
                    if first_chunk and attachment_infos:
                        chunk.attachments = attachment_infos
                        first_chunk = False
                    
                    yield json.dumps(chunk.model_dump(), default=str) + "\n"
                
                # Finalize reasoning and get hash
                reasoning_hash = finalize_reasoning()
                
                # Build metadata with reasoning hash
                msg_metadata = {}
                if reasoning_hash:
                    msg_metadata["reasoning_hash"] = reasoning_hash
                
                # Re-fetch or use captured IDs to avoid detachment
                ai_msg = Message(
                    conversation_id=conv_id,
                    project_id=conv_project_id,
                    org_id=org_id,
                    role=MessageRole.ASSISTANT,
                    agent_id=None,
                    content="".join(full).replace("\x00", ""),
                    status=MessageStatus.SENT,
                    attachments=attachment_infos,
                    metadata_=msg_metadata,
                )
                db.add(ai_msg)
                db.commit()
                db.refresh(ai_msg)
                
                # Send a final chunk with the reasoning hash if available
                if reasoning_hash:
                    yield json.dumps({"reasoning_hash": reasoning_hash}) + "\n"
            except Exception as e:
                error_msg = str(e).lower()
                # Check for rate limit or credit exhaustion errors (Claude/OpenAI)
                if any(x in error_msg for x in ["rate limit", "credit", "quota", "insufficient_quota", "429", "overloaded"]):
                    logger_msg = f"LLM Rate Limit/Quota Error: {str(e)}"
                    print(logger_msg) # Ensure it logs to terminal
                    # Send a structured error message to the client
                    yield json.dumps({
                        "response": "⚠️ **Service Alert**: The AI provider has reached its usage limit or credits are exhausted. Please recharge your Claude/OpenAI credits to continue.",
                        "error": str(e),
                        "type": "error"
                    }) + "\n"
                else:
                    logger_msg = f"LLM Streaming Error: {str(e)}"
                    print(logger_msg)
                    yield json.dumps({
                        "response": f"❌ An error occurred while generating the response: {str(e)}",
                        "error": str(e),
                        "type": "error"
                    }) + "\n"

        return StreamingResponse(stream_cso_agent(), media_type="application/json")

    service = ConversationService(ProviderService.create(user_id=str(current_user.id)))

    async def stream_response():
        try:
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
                attachment_ids=attachment_ids,
            ):
                yield json.dumps(chunk.model_dump(), default=str) + "\n"
        except Exception as e:
            error_msg = str(e).lower()
            if any(x in error_msg for x in ["rate limit", "credit", "quota", "insufficient_quota", "429", "overloaded"]):
                print(f"LLM Rate Limit/Quota Error: {str(e)}")
                yield json.dumps({
                    "response": "⚠️ **Service Alert**: The AI provider has reached its usage limit. Please recharge your Claude/OpenAI credits to continue.",
                    "error": str(e),
                    "type": "error"
                }) + "\n"
            else:
                print(f"LLM Streaming Error: {str(e)}")
                yield json.dumps({
                    "response": f"❌ An error occurred while generating the response: {str(e)}",
                    "error": str(e),
                    "type": "error"
                }) + "\n"

    return StreamingResponse(stream_response(), media_type="application/json")


@router.get("/history/{project_id}")
def get_history(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ConversationService(ProviderService.create(user_id=str(current_user.id)))
    data = service.get_chat_history(db=db, org_id=current_user.org_id, project_id=project_id)
    
    # Ensure all values are JSON-serializable (convert datetime etc.)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__str__') and not isinstance(obj, (str, int, float, bool, type(None))):
            return str(obj)
        return obj
    
    return JSONResponse(content=make_serializable(data))


@router.get("/reasoning/{reasoning_hash}")
async def get_reasoning(
    reasoning_hash: str,
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve saved reasoning content by its SHA-256 hash.
    The hash is returned in the streaming response and stored in message metadata.
    """
    from src.infrastructure.agents.reasoning_manager import load_reasoning_content

    content = load_reasoning_content(reasoning_hash)
    if content is None:
        raise HTTPException(status_code=404, detail="Reasoning content not found")
    return JSONResponse(content={"reasoning_hash": reasoning_hash, "content": content})
