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