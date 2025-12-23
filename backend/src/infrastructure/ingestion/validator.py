from typing import Dict, List
from src.infrastructure.graph.schema import EXTRACTABLE_NODE_TYPES

def validate_segment(segment: Dict) -> Dict:
    valid_entities: List[Dict] = []
    for e in segment.get("entities", []):
        if e.get("type") in EXTRACTABLE_NODE_TYPES and e.get("name"):
            e["source_page"] = segment.get("page_numbers", [])
            e["confidence"] = segment.get("classification_confidence", 0.8)
            e["source_doc_id"] = segment.get("doc_id")
            e["source_version_id"] = segment.get("doc_version")
            valid_entities.append(e)

    index_by_name: Dict[str, str] = {e["name"]: e["type"] for e in valid_entities}

    valid_relationships: List[Dict] = []
    for r in segment.get("relationships", []):
        if r.get("from") and r.get("to") and r.get("type"):
            r["source_page"] = segment.get("page_numbers", [])
            r["source_doc_id"] = segment.get("doc_id")
            r["source_version_id"] = segment.get("doc_version")
            if "from_type" not in r and r.get("from") in index_by_name:
                r["from_type"] = index_by_name[r["from"]]
            if "to_type" not in r and r.get("to") in index_by_name:
                r["to_type"] = index_by_name[r["to"]]
            valid_relationships.append(r)

    segment["entities"] = valid_entities
    segment["relationships"] = valid_relationships
    return segment
