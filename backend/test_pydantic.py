import asyncio
from src.domain.agents.base import ChatContext
try:
    ctx = ChatContext(query="hello", history=[], conversation_id="123", user_id="456")
    print(ctx)
except Exception as e:
    print(f"Error: {e}")
