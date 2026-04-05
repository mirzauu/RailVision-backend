"""
Tool Progress Registry — lightweight pub/sub for real-time tool progress streaming.

Any tool can push progress messages (strings) to its named queue.
The pydantic_agent polls active queues and yields them as PROGRESS events.

Usage in a tool:
    from src.infrastructure.agents.tools.tool_progress import push_progress, begin_tool, end_tool

    async def arun(self, query: str) -> str:
        begin_tool("my_tool_name")
        push_progress("my_tool_name", "🔍 Searching...")
        result = await do_work(query)
        push_progress("my_tool_name", f"✅ Found {len(result)} results")
        end_tool("my_tool_name")
        return result

Usage in pydantic_agent:
    from src.infrastructure.agents.tools.tool_progress import get_queue, DONE_SENTINEL
"""

import asyncio
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Sentinel object pushed to the queue when a tool finishes
DONE_SENTINEL = object()

# Global registry: tool_name → asyncio.Queue
_queues: Dict[str, asyncio.Queue] = {}


def begin_tool(tool_name: str) -> asyncio.Queue:
    """Register a new progress queue for a tool that is starting execution."""
    q: asyncio.Queue = asyncio.Queue()
    _queues[tool_name] = q
    return q


def end_tool(tool_name: str) -> None:
    """Signal that a tool has finished — puts DONE_SENTINEL and removes registry entry."""
    q = _queues.get(tool_name)
    if q is not None:
        try:
            q.put_nowait(DONE_SENTINEL)
        except Exception:
            pass
        _queues.pop(tool_name, None)


def push_progress(tool_name: str, message: str) -> None:
    """Push a progress message for a running tool. Safe to call from sync or async context."""
    q = _queues.get(tool_name)
    if q is None:
        return
    try:
        # Works whether we're inside an event loop or not
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon_threadsafe(q.put_nowait, message)
        else:
            q.put_nowait(message)
    except Exception as exc:
        logger.debug(f"push_progress failed for {tool_name}: {exc}")


async def async_push_progress(tool_name: str, message: str) -> None:
    """Async-friendly version of push_progress."""
    q = _queues.get(tool_name)
    if q is not None:
        await q.put(message)


def get_queue(tool_name: str) -> Optional[asyncio.Queue]:
    """Return the active queue for a tool, or None if not running."""
    return _queues.get(tool_name)


def get_all_active_tool_names() -> list[str]:
    """Return names of all tools currently registered (i.e. in-flight)."""
    return list(_queues.keys())
