from .client import get_index
from .embedder import embed_text
from .writer import upsert_segment, persist_to_pinecone
from .retriever import retrieve_context

__all__ = [
    "get_index",
    "embed_text",
    "upsert_segment",
    "persist_to_pinecone",
    "retrieve_context"
]
