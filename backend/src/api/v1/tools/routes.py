from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.config.database import get_db
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User
from src.api.v1.tools.schemas import ToolInfo
from src.application.tools.service import ToolService

router = APIRouter()


@router.get("/list_tools", response_model=List[ToolInfo])
async def list_tools(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user_id = current_user.id
    tool_service = ToolService(db, user_id)
    return tool_service.list_tools()
