from pinecone import Pinecone
from src.config.settings import settings

pc = Pinecone(api_key=settings.pinecone_api_key)

def get_index():
    return pc.Index(settings.pinecone_index_name)
