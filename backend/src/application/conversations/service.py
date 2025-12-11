from typing import AsyncGenerator, List, Optional
import os
from sqlalchemy.orm import Session
from src.infrastructure.database.models.projects import Project, ProjectAgent, AgentRoleInProject
from src.infrastructure.database.models.agents import Agent
from src.infrastructure.database.models.conversations import Message, Conversation, MessageRole, MessageStatus
from src.infrastructure.llm.provider_service import ProviderService
from src.application.agents.executer_agent import ExecuterAgent
from src.domain.agents.base import AgentConfig, TaskConfig, ChatContext, ChatAgentResponse

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

    def _build_history(self, db: Session, project_id: Optional[str]) -> List[str]:
        if not project_id:
            return []
        msgs = (
            db.query(Message)
            .filter(Message.project_id == project_id)
            .order_by(Message.created_at.asc())
            .limit(20)
            .all()
        )
        return [m.content for m in msgs if m.content]

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
        attachment: Optional[str],
    ) -> ChatAgentResponse:
        # Do not override global model here; routing layer handles model selection safely
        project = db.query(Project).filter(Project.id == project_id).first() if project_id else None
        history = self._build_history(db, project_id)
        resolved_agent = self._resolve_agent(db, project, org_id, agent)
        conv = self._get_or_create_conversation(db, project, org_id)
        user_msg = Message(
            conversation_id=conv.id,
            project_id=conv.project_id,
            org_id=org_id,
            role=MessageRole.USER,
            user_id=user_id,
            content=query,
            status=MessageStatus.SENT,
            attachments=[attachment] if attachment else [],
        )
        db.add(user_msg)
        db.commit()
        db.refresh(user_msg)
        config = self._build_agent_config(resolved_agent)
        ctx = ChatContext(project_id=project_id or "default", history=history, query=query, additional_context=attachment or "")
        agent_runner = ExecuterAgent(self.provider, config, framework=framework or "pydantic")
        resp = await agent_runner.run(ctx)
        ai_msg = Message(
            conversation_id=conv.id,
            project_id=conv.project_id,
            org_id=org_id,
            role=MessageRole.ASSISTANT,
            agent_id=resolved_agent.id if resolved_agent else None,
            content=resp.response,
            status=MessageStatus.SENT,
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
        attachment: Optional[str],
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        # Do not override global model here; routing layer handles model selection safely
        project = db.query(Project).filter(Project.id == project_id).first() if project_id else None
        history = self._build_history(db, project_id)
        resolved_agent = self._resolve_agent(db, project, org_id, agent)
        conv = self._get_or_create_conversation(db, project, org_id)
        user_msg = Message(
            conversation_id=conv.id,
            project_id=conv.project_id,
            org_id=org_id,
            role=MessageRole.USER,
            user_id=user_id,
            content=query,
            status=MessageStatus.SENT,
            attachments=[attachment] if attachment else [],
        )
        db.add(user_msg)
        db.commit()
        db.refresh(user_msg)
        config = self._build_agent_config(resolved_agent)
        ctx = ChatContext(project_id=project_id or "default", history=history, query=query, additional_context=attachment or "")
        agent_runner = ExecuterAgent(self.provider, config, framework=framework or "pydantic")
        full = []
        async for chunk in agent_runner.run_stream(ctx):
            if chunk.response:
                full.append(chunk.response)
            yield chunk
        ai_msg = Message(
            conversation_id=conv.id,
            project_id=conv.project_id,
            org_id=org_id,
            role=MessageRole.ASSISTANT,
            agent_id=resolved_agent.id if resolved_agent else None,
            content="".join(full),
            status=MessageStatus.SENT,
        )
        db.add(ai_msg)
        db.commit()
        db.refresh(ai_msg)

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
        return {
            "conversation_id": conv.id if conv else None,
            "project_id": project_id,
            "messages": msgs,
        }
