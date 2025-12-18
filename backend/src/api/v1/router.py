from fastapi import APIRouter
from src.api.v1.auth import routes as auth_routes
from src.api.v1.organizations import routes as org_routes
from src.api.v1.agents import routes as agent_routes
from src.api.v1.conversations import routes as conversation_routes
from src.api.v1.documents import routes as document_routes
from src.api.v1.integrations import routes as integration_routes
from src.api.v1.memory import routes as memory_routes
from src.api.v1.provider import routes as provider_routes
from src.api.v1.projects import routes as project_routes
from src.api.v1.admin import routes as admin_routes
from src.api.v1.graph import routes as graph_routes

api_router = APIRouter()

api_router.include_router(auth_routes.router, prefix="/auth", tags=["auth"])
api_router.include_router(org_routes.router, prefix="/organizations", tags=["organizations"])
api_router.include_router(agent_routes.router, prefix="/agents", tags=["agents"])
api_router.include_router(conversation_routes.router, prefix="/conversations", tags=["conversations"])
api_router.include_router(document_routes.router, prefix="/documents", tags=["documents"])
api_router.include_router(integration_routes.router, prefix="/integrations", tags=["integrations"])
api_router.include_router(memory_routes.router, prefix="/memory", tags=["memory"])
api_router.include_router(admin_routes.router, prefix="/admin", tags=["admin"])
api_router.include_router(provider_routes.router, prefix="/provider", tags=["provider"])
api_router.include_router(project_routes.router, prefix="/projects", tags=["projects"])
api_router.include_router(graph_routes.router, prefix="/graph", tags=["graph"])
