from src.infrastructure.vector.client import get_index
from src.infrastructure.vector.embedder import embed_text

def retrieve_context(
    query: str,
    doc_id: str = None,
    active_version: str = None,
    allowed_categories: list[str] = None,
    top_k: int = 5
):
    """
    Retrieves execution-safe context from Pinecone.
    
    Args:
        query: The query text
        doc_id: Document ID to filter by
        active_version: Active version of the document
        allowed_categories: List of allowed categories
        top_k: Number of results to return (default 5)
        
    Returns:
        List of dicts with text and score.
    """
    index = get_index()
    vector = embed_text(query)

    filters = {}
    if doc_id:
        filters["doc_id"] = {"$eq": doc_id}
    if active_version:
        filters["doc_version"] = {"$eq": active_version}
    if allowed_categories:
        filters["category"] = {"$in": allowed_categories}

    # Only pass filter if it's not empty, otherwise Pinecone might complain or behave unexpectedly
    query_kwargs = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True
    }
    if filters:
        query_kwargs["filter"] = filters

    result = index.query(**query_kwargs)

    return [
        {
            "text": match["metadata"],
            "score": match["score"]
        }
        for match in result["matches"]
    ]
