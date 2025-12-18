from pathlib import Path
from typing import Dict, Any
import asyncio
from pydantic import BaseModel, Field
from src.infrastructure.llm.provider_service import ProviderService

class _Entity(BaseModel):
    type: str
    name: str
    properties: Dict[str, Any] | None = None

class _Relationship(BaseModel):
    from_: str = Field(alias="from")
    type: str
    to: str

class _ExtractionSchema(BaseModel):
    entities: list[_Entity] = Field(default_factory=list)
    relationships: list[_Relationship] = Field(default_factory=list)

def _load_prompt() -> str:
    p = Path(__file__).parent / "prompts" / "extract.txt"
    return p.read_text(encoding="utf-8")

async def extract_facts(segment: Dict[str, Any]) -> Dict[str, Any]:
    svc = ProviderService.create(user_id="system")
    system = {"role": "system", "content": _load_prompt()}
    user = {"role": "user", "content": segment["text"]}
    try:
        result: _ExtractionSchema = await svc.call_llm_with_structured_output(
            messages=[system, user],
            output_schema=_ExtractionSchema,
            config_type="inference",
        )
        segment["entities"] = [e.model_dump(by_alias=True) for e in result.entities]
        segment["relationships"] = [
            {"from": r.from_, "type": r.type, "to": r.to} for r in result.relationships
        ]
        return segment
    except Exception:
        segment.setdefault("entities", [])
        segment.setdefault("relationships", [])
        return segment
