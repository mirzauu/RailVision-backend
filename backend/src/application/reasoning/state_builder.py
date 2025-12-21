from typing import List, Dict, Any, Optional
from src.infrastructure.graph.neo4j_client import get_neo4j_client

def build_state(doc_ids: Optional[List[str]] = None, query_text: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves the strategic state from Neo4j.
    Filters by doc_ids (from VectorDB matches) or query text keywords.
    Returns a list of dictionaries representing the graph state.
    """
    
    # Base query
    cypher_query = """
    MATCH (p:Product)
    """
    
    where_clauses = []
    params = {}

    if doc_ids:
        where_clauses.append("p.source_doc_id IN $doc_ids")
        params["doc_ids"] = doc_ids
    
    if query_text:
        # Simple fuzzy match on name or description or function
        # Using toLower for case-insensitive search
        where_clauses.append("(toLower(p.name) CONTAINS toLower($query) OR toLower(p.description) CONTAINS toLower($query) OR toLower(p.function) CONTAINS toLower($query))")
        params["query"] = query_text

    if where_clauses:
        cypher_query += " WHERE " + " AND ".join(where_clauses)
    
    # Build the rest of the query
    cypher_query += """
    OPTIONAL MATCH (p)-[:TARGETS]->(c:CustomerSegment)
    OPTIONAL MATCH (p)-[:OPERATES_IN]->(m:Market)
    OPTIONAL MATCH (p)-[:REQUIRES]->(cap:Capability)
    OPTIONAL MATCH (p)-[:LIMITED_BY]->(con:Constraint)
    OPTIONAL MATCH (p)-[:EXPOSED_TO]->(r:Risk)
    RETURN p, collect(c) AS customers,
           collect(m) AS markets,
           collect(cap) AS capabilities,
           collect(con) AS constraints,
           collect(r) AS risks
    LIMIT 10
    """ 
    # Added LIMIT 10 to avoid exploding context if filters are weak.

    client = get_neo4j_client()
    try:
        results = client.run(cypher_query, parameters=params)
        
        state = []
        for row in results:
            state.append({
                "product": row.get("p"),
                "customers": row.get("customers", []),
                "markets": row.get("markets", []),
                "capabilities": row.get("capabilities", []),
                "constraints": row.get("constraints", []),
                "risks": row.get("risks", []),
            })
        return state
    except Exception as e:
        # If query fails, log it (cannot log here easily without importing logger)
        # return empty state
        print(f"Error in build_state: {e}")
        return []
