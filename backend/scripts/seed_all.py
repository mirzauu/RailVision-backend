import os
import sys
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config.database import SessionLocal
from src.infrastructure.database.models.organizations import Organization
from src.infrastructure.database.models.users import User, Role, UserInvitation, RefreshToken, UserStatus, InvitationStatus
from src.infrastructure.database.models.agents import Agent, AgentType, AgentStatus
from src.infrastructure.database.models.projects import Project, ProjectAgent, ProjectMember, ProjectType, ProjectStatus, AgentRoleInProject, MemberRoleInProject
from src.infrastructure.database.models.conversations import Conversation, Message, ConversationType, ConversationStatus, MessageRole, MessageStatus
from src.infrastructure.database.models.documents import Document, DocumentChunk, DocumentType, DocumentStatus, DocumentScope
from src.infrastructure.database.models.integrations import Integration, IntegrationType, IntegrationStatus
from src.infrastructure.database.models.notifications import Notification, NotificationType, NotificationStatus
from src.infrastructure.database.models.logs import LLMUsageLog, AuditLog, AgentCollaborationLog, AuditAction, CollaborationType
from src.shared.security import get_password_hash

def main():
    s = SessionLocal()
    try:
        org = s.query(Organization).filter_by(slug="demo-org").first()
        if org is None:
            org = Organization(name="Demo Org", slug="demo-org")
            s.add(org)
            s.commit()
            s.refresh(org)

        role = s.query(Role).filter_by(name="admin").first()
        if role is None:
            role = Role(name="admin", display_name="Admin", permissions=[], is_system_role=True, is_default=False)
            s.add(role)
            s.commit()
            s.refresh(role)

        user = s.query(User).filter_by(email="demo@example.com").first()
        if user is None:
            hashed = get_password_hash("password123")
            user = User(org_id=org.id, role_id=role.id, email="demo@example.com", hashed_password=hashed, full_name="Demo User", email_verified=True, status=UserStatus.ACTIVE)
            s.add(user)
            s.commit()
            s.refresh(user)

        agent = s.query(Agent).filter_by(name="demo-agent").first()
        if agent is None:
            agent = Agent(org_id=org.id, type=AgentType.GENERAL, name="demo-agent", display_name="Demo Agent", status=AgentStatus.ACTIVE, expertise_areas=[])
            s.add(agent)
            s.commit()
            s.refresh(agent)

        project = s.query(Project).filter_by(name="Demo Project").first()
        if project is None:
            project = Project(org_id=org.id, created_by=user.id, name="Demo Project", type=ProjectType.SINGLE_CHAT, status=ProjectStatus.ACTIVE, deliverables=[], tags=[])
            s.add(project)
            s.commit()
            s.refresh(project)

        pa = s.query(ProjectAgent).filter_by(project_id=project.id, agent_id=agent.id).first()
        if pa is None:
            pa = ProjectAgent(project_id=project.id, agent_id=agent.id, role=AgentRoleInProject.PRIMARY)
            s.add(pa)
            s.commit()
            s.refresh(pa)

        pm = s.query(ProjectMember).filter_by(project_id=project.id, user_id=user.id).first()
        if pm is None:
            pm = ProjectMember(project_id=project.id, user_id=user.id, role=MemberRoleInProject.OWNER)
            s.add(pm)
            s.commit()
            s.refresh(pm)

        conv = s.query(Conversation).filter_by(project_id=project.id).first()
        if conv is None:
            conv = Conversation(project_id=project.id, org_id=org.id, title="Demo Conversation", type=ConversationType.LINEAR, status=ConversationStatus.ACTIVE, participating_agent_ids=[], key_points=[], tags=[])
            s.add(conv)
            s.commit()
            s.refresh(conv)

        msg = s.query(Message).filter_by(conversation_id=conv.id).first()
        if msg is None:
            msg = Message(conversation_id=conv.id, project_id=project.id, org_id=org.id, role=MessageRole.USER, user_id=user.id, content="Hello from demo user", status=MessageStatus.SENT, context_used={})
            s.add(msg)
            s.commit()
            s.refresh(msg)

        doc = s.query(Document).filter_by(filename="demo.txt").first()
        if doc is None:
            doc = Document(org_id=org.id, project_id=project.id, uploaded_by=user.id, filename="demo.txt", original_filename="demo.txt", file_type=DocumentType.TXT, file_size_bytes=12, storage_path="/tmp/demo.txt", status=DocumentStatus.UPLOADED, tags=[], keywords=[], assigned_agent_ids=[], shared_with_project_ids=[])
            s.add(doc)
            s.commit()
            s.refresh(doc)

        chunk = s.query(DocumentChunk).filter_by(document_id=doc.id).first()
        if chunk is None:
            chunk = DocumentChunk(document_id=doc.id, org_id=org.id, project_id=project.id, chunk_index=0, chunk_text="demo content", chunk_tokens=3)
            s.add(chunk)
            s.commit()
            s.refresh(chunk)

        integ = s.query(Integration).filter_by(type=IntegrationType.NOTION).first()
        if integ is None:
            integ = Integration(org_id=org.id, type=IntegrationType.NOTION, name="Notion", status=IntegrationStatus.CONNECTED, connected_by=user.id)
            s.add(integ)
            s.commit()
            s.refresh(integ)

        notif = s.query(Notification).filter_by(user_id=user.id).first()
        if notif is None:
            notif = Notification(org_id=org.id, user_id=user.id, type=NotificationType.SYSTEM_ALERT, title="Welcome", message="Welcome to Demo", status=NotificationStatus.UNREAD)
            s.add(notif)
            s.commit()
            s.refresh(notif)

        inv = s.query(UserInvitation).filter_by(email="invitee@example.com").first()
        if inv is None:
            inv = UserInvitation(org_id=org.id, email="invitee@example.com", role_id=role.id, invited_by=user.id, project_ids=[], token="demo-token", expires_at=datetime.utcnow() + timedelta(days=7), status=InvitationStatus.PENDING)
            s.add(inv)
            s.commit()
            s.refresh(inv)

        rt = s.query(RefreshToken).filter_by(user_id=user.id).first()
        if rt is None:
            rt = RefreshToken(user_id=user.id, token_hash="demo-token-hash", expires_at=datetime.utcnow() + timedelta(days=30))
            s.add(rt)
            s.commit()
            s.refresh(rt)

        llm = s.query(LLMUsageLog).filter_by(user_id=user.id).first()
        if llm is None:
            llm = LLMUsageLog(id=1, org_id=org.id, project_id=project.id, user_id=user.id, agent_id=agent.id, conversation_id=conv.id, message_id=msg.id, provider="openai", model="gpt-4o", prompt_tokens=10, completion_tokens=5, total_tokens=15, status="success")
            s.add(llm)
            s.commit()

        audit = s.query(AuditLog).filter_by(user_id=user.id).first()
        if audit is None:
            audit = AuditLog(id=1, org_id=org.id, project_id=project.id, user_id=user.id, action=AuditAction.MESSAGE_SENT, resource_type="message", resource_id=msg.id, success=True)
            s.add(audit)
            s.commit()

        collab = s.query(AgentCollaborationLog).filter_by(primary_agent_id=agent.id).first()
        if collab is None:
            collab = AgentCollaborationLog(org_id=org.id, project_id=project.id, conversation_id=conv.id, message_id=msg.id, type=CollaborationType.DELEGATION, primary_agent_id=agent.id, collaborating_agent_id=agent.id, task_description="Demo task", success=True)
            s.add(collab)
            s.commit()

        print("seeded-all")
    finally:
        s.close()

if __name__ == "__main__":
    main()
