from typing import List, Dict, Any, Optional
from src.infrastructure.graph.neo4j_client import get_neo4j_client
from src.infrastructure.graph.schema import EXTRACTABLE_NODE_TYPES

def build_state(doc_ids: Optional[List[str]] = None, query_text: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves the strategic state from Neo4j.
    Filters by doc_ids (from VectorDB matches) or query text keywords.
    Returns a list of dictionaries representing the graph state.
    """
    
    # We want to match any extractable node type that matches the filter
    # Since we can't easily do MATCH (n) WHERE labels(n) IN $types efficiently without index hints,
    # and we have specific constraints, we'll try a union approach or a broad match if filtered by doc_id.
    
    # Best approach:
    # 1. If doc_ids is present, we can MATCH (n) WHERE n.source_doc_id IN $doc_ids
    # 2. If query_text is present, we match on name/normalized_name or description
    
    # To avoid scanning the whole DB, we can use the labels.
    
    # We will construct a UNION query for the most common types or just use a generic query if doc_id is constrained.
    # Given the scale might be small per doc, generic match on doc_id is fine.
    
    # However, to be safe and use indexes, we iterate labels.
    
    params = {}
    queries = []
    
    # If we have doc_ids, that's a strong filter.
    if doc_ids:
        params["doc_ids"] = doc_ids
        # We can just match all nodes with this property.
        # Note: Index on source_doc_id would help here.
        queries.append("""
        MATCH (n)
        WHERE n.source_doc_id IN $doc_ids
        RETURN n, labels(n) as lbls
        LIMIT 50
        """)
        
    # If we have query text, we need to search by name/description
    if query_text:
        params["query"] = query_text
        # We'll prioritize name matches
        queries.append("""
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($query) 
           OR toLower(coalesce(n.description, '')) CONTAINS toLower($query)
           OR toLower(coalesce(n.function, '')) CONTAINS toLower($query)
        RETURN n, labels(n) as lbls
        LIMIT 20
        """)

    if not queries:
        return []

    final_query = " UNION ".join(queries)
    
    client = get_neo4j_client()
    try:
        results = client.run(final_query, parameters=params)
        
        state = []
        seen_ids = set()
        
        for row in results:
            node = row.get("n")
            if not node: continue
            
            # Extract properties safely for dict or Node objects
            if isinstance(node, dict):
                props = node
            else:
                props = dict(node)
            
            # Filter public labels
            labels = row.get("lbls", [])
            public_labels = [l for l in labels if l in EXTRACTABLE_NODE_TYPES]
            if not public_labels:
                continue
            primary_label = public_labels[0]
            
            # Dedupe using domain-specific keys
            normalized_name = props.get("normalized_name", props.get("name", "Unknown")).lower().strip()
            source_doc_id = props.get("source_doc_id", "unknown")
            dedupe_key = (primary_label, normalized_name, source_doc_id)
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            
            # Build state entry
            state.append({
                "type": primary_label,
                "name": props.get("name", "Unknown"),
                "normalized_name": normalized_name,
                "source_doc_id": source_doc_id,
                "properties": props
            })
            
        return state
    except Exception as e:
        print(f"Error in build_state: {e}")
        return []
