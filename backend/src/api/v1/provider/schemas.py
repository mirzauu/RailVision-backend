from typing import List, Optional
from pydantic import BaseModel


class ProviderInfo(BaseModel):
    id: str
    name: str
    description: str


class AvailableModelOption(BaseModel):
    id: str
    name: str
    description: str
    provider: str
    is_chat_model: bool
    is_inference_model: bool


class AvailableModelsResponse(BaseModel):
    models: List[AvailableModelOption]


class SetProviderRequest(BaseModel):
    chat_model: Optional[str] = None
    inference_model: Optional[str] = None


class ModelInfo(BaseModel):
    provider: str
    id: str
    name: str


class GetProviderResponse(BaseModel):
    chat_model: Optional[ModelInfo] = None
    inference_model: Optional[ModelInfo] = None


class DualProviderConfig(BaseModel):
    chat_config: GetProviderResponse
    inference_config: GetProviderResponse


class UsageCostRequest(BaseModel):
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    limit: Optional[int] = None
    bucket_width: Optional[str] = "1d"


class CostAmount(BaseModel):
    value: float
    currency: str


class CostItem(BaseModel):
    start_time: int
    end_time: int
    amount: CostAmount
    line_item: Optional[str] = None
    project_id: Optional[str] = None


class UsageCostResponse(BaseModel):
    object: str
    data: List[CostItem]