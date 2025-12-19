from typing import Dict, Any, List
from src.infrastructure.graph.neo4j_client import get_neo4j_client
from src.infrastructure.graph.schema import ALLOWED_NODE_TYPES, ALLOWED_RELATIONSHIPS

def _sanitize_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize properties to ensure they are compatible with Neo4j.
    Neo4j properties must be primitives or lists of primitives.
    This function flattens or stringifies nested dictionaries.
    """
    sanitized = {}
    for k, v in props.items():
        if isinstance(v, (dict, list)) and any(isinstance(i, dict) for i in (v if isinstance(v, list) else [v])):
            # If value is a dict or list containing dicts, serialize to JSON string
            import json
            try:
                sanitized[k] = json.dumps(v)
            except (TypeError, ValueError):
                sanitized[k] = str(v)
        else:
            sanitized[k] = v
    return sanitized

def upsert_node(label: str, key: str, properties: Dict[str, Any]) -> None:
    if label not in ALLOWED_NODE_TYPES:
        raise ValueError(f"Invalid node type: {label}")
    if key not in properties:
        raise ValueError("Key not present in properties")
    
    clean_props = _sanitize_properties(properties)
    
    query = f"""
    MERGE (n:{label} {{ {key}: $key_value }})
    SET n += $properties
    """
    client = get_neo4j_client()
    with client.session() as session:
        session.run(query, key_value=clean_props[key], properties=clean_props)

def upsert_relationship(
    from_label: str,
    from_key: str,
    from_value: Any,
    rel_type: str,
    to_label: str,
    to_key: str,
    to_value: Any,
    metadata: Dict[str, Any],
) -> None:
    # Validation removed to allow dynamic relationship types from LLM
    # if rel_type not in ALLOWED_RELATIONSHIPS:
    #    raise ValueError(f"Invalid relationship type: {rel_type}")
    
    clean_metadata = _sanitize_properties(metadata)
    
    query = f"""
    MATCH (a:{from_label} {{ {from_key}: $from_value }})
    MATCH (b:{to_label} {{ {to_key}: $to_value }})
    MERGE (a)-[r:{rel_type}]->(b)
    SET r += $metadata
    """
    client = get_neo4j_client()
    with client.session() as session:
        session.run(
            query,
            from_value=from_value,
            to_value=to_value,
            metadata=clean_metadata,
        )

def upsert_document(doc_id: str, title: str, doc_type: str) -> None:
    upsert_node(
        label="Document",
        key="doc_id",
        properties={
            "doc_id": doc_id,
            "title": title,
            "doc_type": doc_type,
        },
    )

def upsert_document_version(doc_id: str, version_id: str, hash: str) -> None:
    upsert_node(
        label="DocumentVersion",
        key="version_id",
        properties={
            "version_id": version_id,
            "hash": hash,
            "status": "active",
        },
    )
    upsert_relationship(
        from_label="Document",
        from_key="doc_id",
        from_value=doc_id,
        rel_type="HAS_VERSION",
        to_label="DocumentVersion",
        to_key="version_id",
        to_value=version_id,
        metadata={},
    )

def write_entity(entity: Dict[str, Any], doc_id: str, version_id: str) -> None:
    label = entity["type"]
    props = {
        "name": entity["name"],
        **(entity.get("properties") or {}),
        "source_doc_id": doc_id,
        "source_version_id": version_id,
        "confidence": entity.get("confidence", 0.8),
    }
    upsert_node(label, "name", props)

def write_relationship(rel: Dict[str, Any], doc_id: str, version_id: str) -> None:
    upsert_relationship(
        from_label=rel["from_type"] if "from_type" in rel else rel.get("from_label", "Product"),
        from_key="name",
        from_value=rel["from"],
        rel_type=rel["type"],
        to_label=rel["to_type"] if "to_type" in rel else rel.get("to_label", "Product"),
        to_key="name",
        to_value=rel["to"],
        metadata={
            "source_doc_id": doc_id,
            "source_version_id": version_id,
            "source_pages": rel.get("source_pages", rel.get("source_page", [])),
            "confidence": rel.get("confidence", 0.8),
        },
    )

def persist_to_graph(
    processed_segments: List[Dict[str, Any]],
    doc_id: str,
    version_id: str,
    hash: str,
    title: str,
    doc_type: str,
) -> None:
    upsert_document(doc_id, title, doc_type)
    upsert_document_version(doc_id, version_id, hash)
    for seg in processed_segments:
        for entity in seg.get("entities", []):
            write_entity(entity, doc_id, version_id)
        for rel in seg.get("relationships", []):
            if "from_type" not in rel and "to_type" not in rel:
                continue
            write_relationship(rel, doc_id, version_id)
