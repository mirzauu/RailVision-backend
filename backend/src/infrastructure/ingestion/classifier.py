from pathlib import Path
from typing import Dict, Any, List
import asyncio
from pydantic import BaseModel, Field
from src.infrastructure.llm.provider_service import ProviderService

_CATEGORIES: List[str] = [
    "product",
    "market",
    "pricing",
    "traction",
    "technology",
    "risk",
    "team",
    "financials",
    "other",
]

class _ClassificationSchema(BaseModel):
    category: str = Field(description="Category", examples=["product"])
    confidence: float | None = Field(default=None, description="Confidence 0.0-1.0")

def _load_prompt() -> str:
    p = Path(__file__).parent / "prompts" / "classify.txt"
    return p.read_text(encoding="utf-8")

async def classify_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    svc = ProviderService.create(user_id="system")
    system = {"role": "system", "content": _load_prompt()}
    user = {"role": "user", "content": segment["text"]}
    try:
        result: _ClassificationSchema = await svc.call_llm_with_structured_output(
            messages=[system, user],
            output_schema=_ClassificationSchema,
            config_type="inference",
        )
        cat = (result.category or "other").strip().lower()
        if cat not in _CATEGORIES:
            cat = "other"
        segment["category"] = cat
        segment["classification_confidence"] = result.confidence if result.confidence is not None else 0.5
        return segment
    except Exception:
        segment["category"] = "other"
        segment["classification_confidence"] = 0.5
        return segment
