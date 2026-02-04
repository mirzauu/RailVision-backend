from typing import List,Optional
from fastapi import Depends, APIRouter, HTTPException
from sqlalchemy.orm import Session

from src.config.database import get_db
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User

from src.infrastructure.llm.provider_service import ProviderService
from src.api.v1.provider.schemas import (
    ProviderInfo,
    SetProviderRequest,
    GetProviderResponse,
    AvailableModelsResponse,
    UsageCostRequest,
    UsageCostResponse,
)


router = APIRouter()


@router.get("/list-available-llms/", response_model=List[ProviderInfo])
async def list_available_llms(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        service = ProviderService.create(user_id=user.id)
        return await service.list_available_llms()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing LLM providers: {str(e)}")


@router.get("/list-available-models/", response_model=AvailableModelsResponse)
async def list_available_models(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        service = ProviderService.create(user_id=user.id)
        return await service.list_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing available models: {str(e)}")


@router.post("/set-global-ai-provider/")
async def set_global_ai_provider(
    provider_request: SetProviderRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        service = ProviderService.create(user_id=user.id)
        return await service.set_global_ai_provider(provider_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting global AI provider: {str(e)}")


@router.get("/get-global-ai-provider/", response_model=GetProviderResponse)
async def get_global_ai_provider(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        service = ProviderService.create(user_id=user.id)
        return await service.get_global_ai_provider()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting global AI provider: {str(e)}")


@router.get("/openai-usage/", response_model=UsageCostResponse)
async def get_openai_usage(
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None,
    bucket_width: Optional[str] = "1d",
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        service = ProviderService.create(user_id=user.id)
        request = UsageCostRequest(
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            bucket_width=bucket_width
        )
        return await service.get_openai_usage_costs(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting OpenAI usage: {str(e)}")