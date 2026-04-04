from typing import AsyncGenerator, List, Optional, Dict
import os
from sqlalchemy.orm import Session
from src.infrastructure.database.models.projects import Project, ProjectAgent, AgentRoleInProject
from src.infrastructure.database.models.agents import Agent
from src.infrastructure.database.models.conversations import Message, Conversation, MessageRole, MessageStatus
from src.infrastructure.database.models.ppt import Presentation
from src.infrastructure.database.models.generated_pdf import GeneratedPDF
from src.infrastructure.database.models.generated_word import GeneratedWord
from src.infrastructure.database.models.documents import Document
from src.infrastructure.llm.provider_service import ProviderService
from src.application.agents.executer_agent import ExecuterAgent
from src.domain.agents.base import AgentConfig, TaskConfig, ChatContext, ChatAgentResponse
from src.application.tools.service import ToolService
from src.infrastructure.agents.reasoning_manager import reset_reasoning_manager, finalize_reasoning

class ConversationService:
    def __init__(self, provider: ProviderService):
        self.provider = provider

    def _resolve_agent(self, db: Session, project: Optional[Project], org_id: str, agent_hint: Optional[str]) -> Optional[Agent]:
        if agent_hint:
            a = db.query(Agent).filter(Agent.id == agent_hint).first()
            if a:
                return a
        if project:
            pa = db.query(ProjectAgent).filter(ProjectAgent.project_id == project.id, ProjectAgent.role == AgentRoleInProject.PRIMARY).first()
            if pa:
                a = db.query(Agent).filter(Agent.id == pa.agent_id).first()
                if a:
                    return a
        a = db.query(Agent).filter(Agent.org_id == org_id).first()
        return a

    def _build_history(self, db: Session, project_id: Optional[str]) -> List[Dict[str, str]]:
        from src.infrastructure.agents.reasoning_manager import load_reasoning_content

        if not project_id:
            return []
        msgs = (
            db.query(Message)
            .filter(Message.project_id == project_id)
            .order_by(Message.created_at.asc())
            .limit(20)
            .all()
        )
        history = []
        for m in msgs:
            if m.content:
                role = "user" if m.role == MessageRole.USER else "assistant"
                content = m.content
                if role == "assistant" and getattr(m, "metadata_", None) and isinstance(m.metadata_, dict) and "reasoning_hash" in m.metadata_:
                    reasoning_hash = m.metadata_["reasoning_hash"]
                    reasoning = load_reasoning_content(reasoning_hash)
                    print(f"\n--- REASONING HASH: {reasoning_hash} ---")
                    print(f"--- REASONING DATA: {reasoning} ---\n")
                    if reasoning:
                        content = f"<reasoning>\n{reasoning}\n</reasoning>\n{content}"
                history.append({"role": role, "content": content})
        return history

    def _get_or_create_conversation(self, db: Session, project: Optional[Project], org_id: str) -> Conversation:
        if project:
            conv = db.query(Conversation).filter(Conversation.project_id == project.id).first()
            if conv:
                return conv
            conv = Conversation(project_id=project.id, org_id=org_id, title="Conversation")
            db.add(conv)
            db.commit()
            db.refresh(conv)
            return conv
        conv = db.query(Conversation).filter(Conversation.org_id == org_id).first()
        if conv:
            return conv
        raise RuntimeError("conversation requires a valid project")

    def _build_agent_config(self, agent: Optional[Agent]) -> AgentConfig:
        role = agent.display_name if agent and agent.display_name else "General Agent"
        goal = "Answer the query"
        backstory = agent.description or "Assistant"
        return AgentConfig(
            role=role,
            goal=goal,
            backstory=backstory,
            tasks=[
                TaskConfig(
                    description="Answer the user's question using available context",
                    expected_output="Clear, concise answer with any relevant references",
                )
            ],
        )

    async def chat(
        self,
        db: Session,
        user_id: str,
        org_id: str,
        query: str,
        project_id: Optional[str],
        framework: Optional[str],
        model: Optional[str],
        agent: Optional[str],
        attachment: Optional[str] = None,
        attachment_ids: Optional[List[str]] = None,
    ) -> ChatAgentResponse:
        # Do not override global model here; routing layer handles model selection safely
        project = db.query(Project).filter(Project.id == project_id).first() if project_id else None
        history = self._build_history(db, project_id)
        resolved_agent = self._resolve_agent(db, project, org_id, agent)
        conv = self._get_or_create_conversation(db, project, org_id)
        
        attachment_infos = []
        if attachment_ids:
            documents = db.query(Document).filter(Document.id.in_(attachment_ids)).all()
            for doc in documents:
                attachment_infos.append({
                    "id": doc.id,
                    "filename": doc.original_filename,
                    "file_type": str(doc.file_type),
                    "file_size_bytes": doc.file_size_bytes,
                    "status": str(doc.status)
                })

        user_msg = Message(
            conversation_id=conv.id,
            project_id=conv.project_id,
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
        config = self._build_agent_config(resolved_agent)
        
        ctx = ChatContext(history=history, query=query, additional_context=attachment or "")
        tool_service = ToolService(db, user_id, conversation_id=conv.id)
        agent_runner = ExecuterAgent(self.provider, config, framework=framework or "pydantic", tools_provider=tool_service)
        resp = await agent_runner.run(ctx)
        if attachment_infos:
            resp.attachments = attachment_infos
        ai_msg = Message(
            conversation_id=conv.id,
            project_id=conv.project_id,
            org_id=org_id,
            role=MessageRole.ASSISTANT,
            agent_id=resolved_agent.id if resolved_agent else None,
            content=resp.response.replace("\x00", ""),
            status=MessageStatus.SENT,
            attachments=attachment_infos,
        )
        db.add(ai_msg)
        db.commit()
        db.refresh(ai_msg)
        return resp

    async def chat_stream(
        self,
        db: Session,
        user_id: str,
        org_id: str,
        query: str,
        project_id: Optional[str],
        framework: Optional[str],
        model: Optional[str],
        agent: Optional[str],
        attachment: Optional[str] = None,
        attachment_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        # Do not override global model here; routing layer handles model selection safely
        project = db.query(Project).filter(Project.id == project_id).first() if project_id else None
        history = self._build_history(db, project_id)
        resolved_agent = self._resolve_agent(db, project, org_id, agent)
        conv = self._get_or_create_conversation(db, project, org_id)

        attachment_infos = []
        if attachment_ids:
            documents = db.query(Document).filter(Document.id.in_(attachment_ids)).all()
            for doc in documents:
                attachment_infos.append({
                    "id": doc.id,
                    "filename": doc.original_filename,
                    "file_type": str(doc.file_type),
                    "file_size_bytes": doc.file_size_bytes,
                    "status": str(doc.status)
                })

        user_msg = Message(
            conversation_id=conv.id,
            project_id=conv.project_id,
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
        config = self._build_agent_config(resolved_agent)
        
        ctx = ChatContext(history=history, query=query, additional_context=attachment or "")
        tool_service = ToolService(db, user_id, conversation_id=conv.id)
        agent_runner = ExecuterAgent(self.provider, config, framework=framework or "pydantic", tools_provider=tool_service)
        full = []
        print("Agent running...", query)
        first_chunk = True
        reset_reasoning_manager()
        async for chunk in agent_runner.run_stream(ctx):
            if chunk.response:
                full.append(chunk.response)
            if first_chunk and attachment_infos:
                chunk.attachments = attachment_infos
                first_chunk = False
            yield chunk
        
        # Finalize reasoning and get hash
        reasoning_hash = finalize_reasoning()
        
        # Build metadata with reasoning hash
        msg_metadata = {}
        if reasoning_hash:
            msg_metadata["reasoning_hash"] = reasoning_hash
        
        ai_msg = Message(
            conversation_id=conv.id,
            project_id=conv.project_id,
            org_id=org_id,
            role=MessageRole.ASSISTANT,
            agent_id=resolved_agent.id if resolved_agent else None,
            content="".join(full).replace("\x00", ""),
            status=MessageStatus.SENT,
            attachments=attachment_infos,
            metadata_=msg_metadata,
        )
        db.add(ai_msg)
        db.commit()
        db.refresh(ai_msg)
        
        # Yield a final chunk with the reasoning hash if available
        if reasoning_hash:
            yield ChatAgentResponse(
                response="",
                tool_calls=[],
                citations=[],
                reasoning_hash=reasoning_hash,
            )

    def get_chat_history(self, db: Session, org_id: str, project_id: str) -> dict:
        conv = (
            db.query(Conversation)
            .filter(Conversation.project_id == project_id, Conversation.org_id == org_id)
            .first()
        )
        msgs = (
            db.query(Message)
            .filter(Message.project_id == project_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        
        # Fetch presentations link to this conversation
        presentations = []
        generated_pdfs = []
        generated_word_docs = []
        if conv:
            presentations = (
                db.query(Presentation)
                .filter(Presentation.conversation_id == conv.id)
                .all()
            )
            generated_pdfs = (
                db.query(GeneratedPDF)
                .filter(GeneratedPDF.conversation_id == conv.id)
                .all()
            )
            generated_word_docs = (
                db.query(GeneratedWord)
                .filter(GeneratedWord.conversation_id == conv.id)
                .all()
            )

        # Convert SQLAlchemy models to dictionaries
        def model_to_dict(obj):
            if obj is None:
                return None
            return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

        # Enhanced serialization helper
        def serialize_presentation(ppt):
            if not ppt: return None
            data = model_to_dict(ppt)
            data['slides'] = [model_to_dict(s) for s in ppt.slides]
            return data

        def serialize_pdf(pdf):
            if not pdf: return None
            data = model_to_dict(pdf)
            data['sections'] = [model_to_dict(s) for s in pdf.sections]
            return data

        def serialize_word(word):
            if not word: return None
            data = model_to_dict(word)
            data['sections'] = [model_to_dict(s) for s in word.sections]
            return data

        return {
            "conversation_id": conv.id if conv else None,
            "project_id": project_id,
            "messages": [model_to_dict(m) for m in msgs],
            "presentations": [serialize_presentation(p) for p in presentations],
            "generated_pdfs": [serialize_pdf(p) for p in generated_pdfs],
            "generated_word_docs": [serialize_word(w) for w in generated_word_docs]
        }

