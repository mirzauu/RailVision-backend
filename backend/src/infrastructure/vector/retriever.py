from src.infrastructure.vector.client import get_index
from src.infrastructure.vector.embedder import embed_text

def retrieve_context(
    query: str,
    doc_id: str,
    active_version: str,
    allowed_categories: list[str],
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

    filters = {
        "doc_id": {"$eq": doc_id},
        "doc_version": {"$eq": active_version},
        "category": {"$in": allowed_categories}
    }

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filters
    )

    return [
        {
            "text": match["metadata"],
            "score": match["score"]
        }
        for match in result["matches"]
    ]
