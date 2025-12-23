from typing import Dict, Any, List
from collections import defaultdict
from src.infrastructure.graph.neo4j_client import get_neo4j_client
from src.infrastructure.graph.schema import ALLOWED_NODE_TYPES
from src.infrastructure.ingestion.resolution import normalize_name

def _sanitize_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize properties to ensure they are compatible with Neo4j.
    Neo4j properties must be primitives or lists of primitives.
    This function flattens or stringifies nested dictionaries.
    """
    sanitized = {}
    for k, v in props.items():
        if isinstance(v, (dict, list)) and any(isinstance(i, dict) for i in (v if isinstance(v, list) else [v])):
            import json
            try:
                sanitized[k] = json.dumps(v)
            except (TypeError, ValueError):
                sanitized[k] = str(v)
        else:
            sanitized[k] = v
    return sanitized

def persist_to_graph(
    processed_segments: List[Dict[str, Any]],
    doc_id: str,
    version_id: str,
    hash: str,
    title: str,
    doc_type: str,
) -> None:
    client = get_neo4j_client()
    with client.session() as session:
        # 1. Document & Version
        session.run("""
        MERGE (d:Document {doc_id: $doc_id})
        SET d.title = $title, d.doc_type = $doc_type
        MERGE (v:DocumentVersion {version_id: $version_id})
        SET v.hash = $hash, v.status = 'active'
        MERGE (d)-[:HAS_VERSION]->(v)
        """, doc_id=doc_id, version_id=version_id, title=title, doc_type=doc_type, hash=hash)

        # 2. Collect Data
        # Key: (label, normalized_name) -> {props, segment_ids, versions}
        entity_map = {}
        # Key: (from_label, from_norm, to_label, to_norm, rel_type) -> {props, segment_ids, versions}
        rels_map = {}

        for seg in processed_segments:
            seg_id = seg.get("segment_id")
            
            # Process Entities
            for entity in seg.get("entities", []):
                label = entity.get("type")
                if label not in ALLOWED_NODE_TYPES: continue
                
                raw_name = entity.get("name")
                if not raw_name: continue
                
                norm_name = normalize_name(raw_name)
                key = (label, norm_name)
                
                props = {
                    "name": raw_name,
                    "normalized_name": norm_name,
                    **(entity.get("properties") or {}),
                    "confidence": entity.get("confidence", 0.8),
                    "source_doc_id": doc_id # Keep latest doc ref
                }
                
                if key not in entity_map:
                    entity_map[key] = {
                        "props": props,
                        "segment_ids": {seg_id} if seg_id else set(),
                        "source_versions": {version_id}
                    }
                else:
                    entity_map[key]["props"].update(props)
                    if seg_id:
                        entity_map[key]["segment_ids"].add(seg_id)
                    entity_map[key]["source_versions"].add(version_id)
            
            # Process Relationships
            for rel in seg.get("relationships", []):
                from_label = rel.get("from_type")
                to_label = rel.get("to_type")
                rel_type = rel.get("type")
                
                if not from_label or not to_label or not rel_type: continue
                
                from_raw = rel.get("from")
                to_raw = rel.get("to")
                
                if not from_raw or not to_raw: continue
                
                from_norm = normalize_name(from_raw)
                to_norm = normalize_name(to_raw)
                
                # Ensure implicit nodes exist
                for (lbl, raw, norm) in [(from_label, from_raw, from_norm), (to_label, to_raw, to_norm)]:
                    if lbl in ALLOWED_NODE_TYPES:
                        e_key = (lbl, norm)
                        if e_key not in entity_map:
                            entity_map[e_key] = {
                                "props": {
                                    "name": raw, 
                                    "normalized_name": norm, 
                                    "implicit": True,
                                    "source_doc_id": doc_id
                                },
                                "segment_ids": {seg_id} if seg_id else set(),
                                "source_versions": {version_id}
                            }
                        else:
                            if seg_id:
                                entity_map[e_key]["segment_ids"].add(seg_id)
                            entity_map[e_key]["source_versions"].add(version_id)

                # Relationship Data
                r_key = (from_label, from_norm, to_label, to_norm, rel_type)
                r_props = {
                    "confidence": rel.get("confidence", 0.8),
                    "from_normalized_name": from_norm,
                    "to_normalized_name": to_norm,
                    "source_doc_id": doc_id
                }
                
                if r_key not in rels_map:
                    rels_map[r_key] = {
                        "props": r_props,
                        "segment_ids": {seg_id} if seg_id else set(),
                        "source_versions": {version_id}
                    }
                else:
                    rels_map[r_key]["props"].update(r_props)
                    if seg_id:
                        rels_map[r_key]["segment_ids"].add(seg_id)
                    rels_map[r_key]["source_versions"].add(version_id)

        # 3. Batch Upsert Entities
        nodes_by_label = defaultdict(list)
        for (label, _), data in entity_map.items():
            final_props = _sanitize_properties(data["props"])
            final_props["segment_ids"] = list(data["segment_ids"])
            final_props["source_versions"] = list(data["source_versions"])
            nodes_by_label[label].append(final_props)

        for label, nodes in nodes_by_label.items():
            if not nodes: continue
            # Note: We use normalized_name as the merge key
            query = f"""
            UNWIND $batch as row
            MERGE (n:{label} {{normalized_name: row.normalized_name}})
            ON CREATE SET n.created_at = timestamp()
            SET n += row
            """
            session.run(query, batch=nodes)

        # 4. Batch Upsert Relationships
        rels_by_type = defaultdict(list)
        for (from_l, from_n, to_l, to_n, r_type), data in rels_map.items():
            final_props = _sanitize_properties(data["props"])
            final_props["segment_ids"] = list(data["segment_ids"])
            final_props["source_versions"] = list(data["source_versions"])
            
            group_key = (from_l, to_l, r_type)
            rels_by_type[group_key].append(final_props)

        for (from_label, to_label, rel_type), rels in rels_by_type.items():
            if not rels: continue
            query = f"""
            UNWIND $batch as row
            MATCH (a:{from_label} {{normalized_name: row.from_normalized_name}})
            MATCH (b:{to_label} {{normalized_name: row.to_normalized_name}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += row
            """
            session.run(query, batch=rels)
