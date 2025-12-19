from src.infrastructure.vector.client import get_index
from src.infrastructure.vector.embedder import embed_text

def upsert_segment(
    segment_id: str,
    text: str,
    metadata: dict
):
    index = get_index()
    vector = embed_text(text)

    # Pinecone metadata requirement: lists must be of strings, not numbers.
    if "page_numbers" in metadata and isinstance(metadata["page_numbers"], list):
        metadata["page_numbers"] = [str(p) for p in metadata["page_numbers"]]

    index.upsert([
        {
            "id": segment_id,
            "values": vector,
            "metadata": metadata
        }
    ])

def persist_to_pinecone(processed_segments: list[dict]):
    """
    Persists processed segments to Pinecone.
    
    Args:
        processed_segments: List of dicts containing segment data.
        Each segment dict should have:
        - doc_id
        - doc_version 
        - segment_id
        - text
        - category
        - page_numbers
        - classification_confidence (optional, defaults to 0.8)
    """
    for seg in processed_segments:
        segment_id = f"{seg['doc_id']}:{seg['doc_version']}:{seg.get('segment_id', '')}"
        
        # Ensure we have a segment_id if not in dict
        if not seg.get('segment_id'):
             # fallback or error handling logic if needed, but assuming unique ID construction strategy
             pass 

        upsert_segment(
            segment_id=segment_id,
            text=seg["text"],
            metadata={
                "doc_id": seg["doc_id"],
                "doc_version": seg["doc_version"],
                "category": seg["category"],
                "page_numbers": seg["page_numbers"],
                "confidence": seg.get("classification_confidence", 0.8)
            }
        )
