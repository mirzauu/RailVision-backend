"""
Shared Claude Document Generator — core logic for generating .pptx / .docx files
via Claude's code execution skills (beta).

This module eliminates code duplication between ppt_tool.py and word_tool.py by
providing a single, well-tested abstraction for:
  1. Calling Claude with code-execution + file skills
  2. Extracting file IDs from structured response content blocks
  3. Downloading and writing the binary file to disk
  4. Hybrid approach: streaming first pass, then multi-turn retry if needed
  5. Real-time progress streaming via asyncio.Queue

Usage:
    # Simple (no streaming):
    success, text = await generate_document(
        prompt="...",
        output_path="/abs/path/to/file.pptx",
        skill_id="pptx",
        file_extension=".pptx",
    )

    # With real-time progress:
    queue = asyncio.Queue()
    task = asyncio.create_task(generate_document(
        prompt="...",
        output_path="/abs/path/to/file.pptx",
        skill_id="pptx",
        file_extension=".pptx",
        progress_queue=queue,
    ))
    async for chunk in drain_queue(queue):
        print(chunk)  # real-time text from Claude
    success, text = await task
"""

import os
import re
import logging
import asyncio
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel to signal "queue is done"
# ---------------------------------------------------------------------------
_QUEUE_DONE = object()

# ---------------------------------------------------------------------------
# Singleton Anthropic client (reuses HTTP connection pool)
# ---------------------------------------------------------------------------
_anthropic_client = None


def _get_client():
    """Return a cached Anthropic client instance."""
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        from src.config.settings import settings

        api_key = settings.claude_api_key or os.environ.get("CLAUDE_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if not api_key:
            raise RuntimeError(
                "Neither CLAUDE_API_KEY nor ANTHROPIC_API_KEY is set in the environment."
            )
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


# ---------------------------------------------------------------------------
# Structured file-ID extraction (no more str() + regex hacks)
# ---------------------------------------------------------------------------

_BETAS = [
    "code-execution-2025-08-25",
    "skills-2025-10-02",
    "files-api-2025-04-14",
]
_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 8192


def _extract_file_ids_from_content(content_blocks) -> list[str]:
    """Extract file IDs from structured response content blocks.

    Looks for content blocks with a ``file_id`` attribute (code execution
    results that produce files) instead of regex-matching a stringified object.
    Falls back to regex on block str() only as a last resort.
    """
    file_ids: list[str] = []

    for block in content_blocks:
        # 1) Structured extraction — content block types that carry file info
        #    e.g. type="code_execution_result" with content[].file_id
        if hasattr(block, "content") and isinstance(block.content, list):
            for sub in block.content:
                fid = getattr(sub, "file_id", None)
                if fid and fid not in file_ids:
                    logger.info(f"[STRUCTURED] Captured file_id: {fid}")
                    file_ids.append(fid)

        # Direct file_id on the block itself
        fid = getattr(block, "file_id", None)
        if fid and fid not in file_ids:
            logger.info(f"[STRUCTURED] Captured file_id: {fid}")
            file_ids.append(fid)

    # 2) Fallback: regex over str(content_blocks) — only if structured failed
    if not file_ids:
        blob = str(content_blocks)
        matches = re.findall(r"file_[a-zA-Z0-9]{15,}", blob)
        for fid in matches:
            if fid not in file_ids:
                logger.info(f"[REGEX FALLBACK] Captured file_id: {fid}")
                file_ids.append(fid)

    return file_ids


# ---------------------------------------------------------------------------
# File download + selection
# ---------------------------------------------------------------------------


def _select_and_download(
    client,
    file_ids: list[str],
    output_path: str,
    file_extension: str,
) -> bool:
    """Pick the best file (matching extension), download it, and validate."""
    if not file_ids:
        return False

    # Prefer the file whose metadata matches the desired extension
    selected_id = file_ids[-1]  # default: last one
    for fid in reversed(file_ids):
        try:
            meta = client.beta.files.retrieve_metadata(fid, betas=_BETAS)
            if meta.filename and meta.filename.lower().endswith(file_extension):
                selected_id = fid
                logger.info(
                    f"Selected file by extension: {meta.filename} ({fid})"
                )
                break
        except Exception as exc:
            logger.warning(f"Could not retrieve metadata for {fid}: {exc}")

    logger.info(f"Downloading file_id: {selected_id} → {output_path}")

    file_content = client.beta.files.download(
        file_id=selected_id, betas=_BETAS
    )

    # Write binary
    with open(output_path, "wb") as fp:
        if hasattr(file_content, "content"):
            fp.write(file_content.content)
        elif hasattr(file_content, "read"):
            fp.write(file_content.read())
        else:
            fp.write(file_content)

    # Validate — .pptx and .docx are ZIP-based formats
    if not zipfile.is_zipfile(output_path):
        logger.error(f"Downloaded file is not a valid ZIP archive: {output_path}")
        os.remove(output_path)
        return False

    logger.info(f"File saved and validated: {output_path}")
    return True


# ---------------------------------------------------------------------------
# Core generator (hybrid: streaming first, multi-turn retry)
# ---------------------------------------------------------------------------


def _run_claude_sync(
    prompt: str,
    output_path: str,
    skill_id: str,
    file_extension: str,
    max_retries: int = 2,
    progress_callback=None,
) -> tuple[bool, str]:
    """Call Claude with code execution to produce a document file.

    Strategy:
      1. First attempt uses streaming for fast, real-time file-ID capture.
      2. If streaming doesn't yield a file, falls back to a non-streaming
         multi-turn loop (up to ``max_retries`` continuation turns).

    Args:
        progress_callback: Optional callable(str) invoked for each text chunk
                          in real-time during streaming. Thread-safe.

    Returns (success: bool, text_output: str).
    """
    client = _get_client()
    all_file_ids: list[str] = []
    collected_text: list[str] = []

    common_kwargs = dict(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        betas=_BETAS,
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        extra_body={
            "container": {
                "skills": [
                    {"type": "anthropic", "skill_id": skill_id, "version": "latest"}
                ]
            }
        },
    )

    def _emit(text: str):
        """Push a text chunk to the progress callback if provided."""
        if progress_callback and text:
            try:
                progress_callback(text)
            except Exception:
                pass

    # ---- Pass 1: Streaming ----
    try:
        logger.info(f"[Pass 1 — Stream] Generating {file_extension} with skill={skill_id}")
        with client.beta.messages.stream(
            messages=[{"role": "user", "content": prompt}],
            **common_kwargs,
        ) as stream:
            for event in stream:
                # Collect and emit text deltas in real-time
                if (
                    event.type == "content_block_delta"
                    and hasattr(event.delta, "text")
                ):
                    chunk = event.delta.text
                    collected_text.append(chunk)
                    _emit(chunk)

            # After streaming completes, get the final message for structured extraction
            final_message = stream.get_final_message()
            all_file_ids = _extract_file_ids_from_content(final_message.content)

    except Exception as exc:
        logger.warning(f"[Pass 1 — Stream] Error during streaming: {exc}")

    # If streaming found file IDs, download immediately
    if all_file_ids:
        _emit("\n\n📦 Downloading generated file…")
        ok = _select_and_download(client, all_file_ids, output_path, file_extension)
        if ok:
            _emit("\n✅ File downloaded and validated!")
            return True, "".join(collected_text)

    # ---- Pass 2: Multi-turn retry (non-streaming) ----
    logger.info(f"[Pass 2 — Retry] Streaming did not yield a file. Trying multi-turn…")
    _emit("\n\n🔄 Retrying document generation…")
    messages_history = [{"role": "user", "content": prompt}]
    all_file_ids.clear()
    collected_text.clear()

    for turn in range(max_retries + 1):  # initial + retries
        logger.info(f"[Pass 2 — Turn {turn + 1}] Sending request…")
        _emit(f"\n⏳ Attempt {turn + 1}…")
        try:
            response = client.beta.messages.create(
                messages=messages_history,
                **common_kwargs,
            )
        except Exception as exc:
            logger.error(f"[Pass 2 — Turn {turn + 1}] API error: {exc}")
            break

        # Extract file IDs from structured content
        ids = _extract_file_ids_from_content(response.content)
        for fid in ids:
            if fid not in all_file_ids:
                all_file_ids.append(fid)

        # Capture text and emit it
        for block in response.content:
            if getattr(block, "type", "") == "text":
                collected_text.append(block.text)
                _emit(block.text)

        # Append assistant response for potential continuation
        messages_history.append({"role": "assistant", "content": response.content})

        if all_file_ids:
            logger.info(f"[Pass 2] File ID(s) found — stopping loop.")
            break

        if response.stop_reason == "end_turn":
            logger.info(f"[Pass 2] Model ended turn without producing a file.")
            break

        logger.info(
            f"[Pass 2] stop_reason={response.stop_reason}, continuing…"
        )

    text_output = "".join(collected_text)

    if all_file_ids:
        _emit("\n\n📦 Downloading generated file…")
        ok = _select_and_download(client, all_file_ids, output_path, file_extension)
        if ok:
            _emit("\n✅ File downloaded and validated!")
            return True, text_output

    logger.error("No file captured from Claude after all attempts.")
    return False, text_output


# ---------------------------------------------------------------------------
# Public async wrapper with real-time progress queue
# ---------------------------------------------------------------------------


async def generate_document(
    prompt: str,
    output_path: str,
    skill_id: str,
    file_extension: str,
    max_retries: int = 2,
    progress_queue: Optional[asyncio.Queue] = None,
) -> tuple[bool, str]:
    """Async wrapper — runs the synchronous Claude call in a thread.

    Args:
        prompt:         Full prompt including document content instructions.
        output_path:    Absolute path where the generated file will be saved.
        skill_id:       Claude skill to activate ("pptx" or "docx").
        file_extension: Expected file extension (e.g. ".pptx", ".docx").
        max_retries:    Max continuation turns in the retry pass (default 2).
        progress_queue: Optional asyncio.Queue to receive real-time text chunks.
                       When provided, text chunks are pushed as they arrive
                       from Claude's streaming response. A sentinel is sent
                       when done.

    Returns:
        (success, claude_text_output)
    """
    loop = asyncio.get_event_loop()

    def _progress_callback(text: str):
        """Thread-safe: schedule put onto the async queue from the sync thread."""
        if progress_queue is not None:
            loop.call_soon_threadsafe(progress_queue.put_nowait, text)

    callback = _progress_callback if progress_queue is not None else None

    try:
        result = await asyncio.to_thread(
            _run_claude_sync,
            prompt,
            output_path,
            skill_id,
            file_extension,
            max_retries,
            callback,
        )
        return result
    finally:
        # Signal the queue consumer that we're done
        if progress_queue is not None:
            progress_queue.put_nowait(_QUEUE_DONE)
