from typing import Optional
from supabase import create_client, Client
from src.config.settings import settings

_client: Optional[Client] = None

def get_supabase() -> Client:
    global _client
    if _client is None:
        if not settings.supabase_url or not settings.supabase_anon_key:
            raise RuntimeError("Supabase URL or ANON KEY not configured in environment")
        _client = create_client(settings.supabase_url, settings.supabase_anon_key)
    return _client
