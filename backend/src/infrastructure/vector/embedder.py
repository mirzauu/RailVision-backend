from typing import List
from openai import OpenAI
from src.config.settings import settings

client = OpenAI(api_key=settings.openai_api_key)

EMBEDDING_MODEL = "text-embedding-3-large"

def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    # Remove newlines to improve performance/quality as suggested by OpenAI sometimes, though less critical for v3
    # But crucial: Handle empty strings to avoid API errors
    sanitized = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=sanitized
    )
    return [d.embedding for d in response.data]
