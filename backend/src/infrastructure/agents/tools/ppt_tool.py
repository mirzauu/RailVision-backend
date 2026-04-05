"""
PowerPoint Tool — generates professional .pptx files using Claude's code execution skill.

Architecture:
  User → Agent → create_ppt tool
                    ↓
                SlideData (title + content)
                    ↓
                Claude (Generates PPT with code_execution)
                    ↓
                storage/presentations/<name>_<uuid>.pptx
                    ↓
                Download URL

Supports real-time progress streaming via asyncio.Queue for word-by-word
updates during document generation.
"""

import os
import re
import uuid
import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from src.infrastructure.database.models import User, Presentation, PresentationSlide
from src.infrastructure.agents.tools.claude_document_generator import (
    generate_document,
    _QUEUE_DONE,
)
from src.config.settings import settings

logger = logging.getLogger(__name__)

# Limits
MAX_SLIDES = 30
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Absolute storage root (derived from project base, not CWD)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # …/backend
STORAGE_DIR = _PROJECT_ROOT / "storage" / "presentations"

# Global registry for active progress queues, keyed by tool call context
# This allows the pydantic agent to drain progress while the tool runs
_active_progress_queues: Dict[str, asyncio.Queue] = {}


def get_active_progress_queue(tool_name: str) -> Optional[asyncio.Queue]:
    """Get the active progress queue for a running tool, if any."""
    return _active_progress_queues.get(tool_name)


class PptSlideInput(BaseModel):
    """Represents one slide in the presentation."""
    title: str = Field(description="Title for this slide.")
    content: str = Field(
        description="Content for this slide. Supports standard markdown, bullet points, numbered items, and tables."
    )
    slide_type: str = Field(
        default="bullet",
        description="Type of slide: 'bullet', 'text', 'title', 'two_column'.",
    )


class CreatePptInput(BaseModel):
    """Input schema for the create_ppt tool."""
    title: str = Field(description="Title of the PowerPoint presentation.")
    slides: List[PptSlideInput] = Field(
        description="List of slides. Each has 'title', 'content', and optional 'slide_type'."
    )


def _build_prompt(title: str, slides: List[PptSlideInput]) -> str:
    """Build the Claude prompt from structured slide data."""
    parts = [
        f"Generate a highly professional PowerPoint presentation titled '{title}'.",
        "Use python-pptx to generate the presentation, save it, and ensure we get the file down.\n",
        "Slide Content:",
    ]
    for idx, slide in enumerate(slides):
        parts.append(f"--- Slide {idx + 1} ---")
        parts.append(f"Slide Type: {slide.slide_type}")
        parts.append(f"Title: {slide.title}")
        parts.append(f"Content:\n{slide.content}\n")

    parts.append(
        "\nIMPORTANT: Please generate only the .pptx presentation file. "
        "Avoid creating summary text files or extra reports in the environment "
        "to ensure the correct file is captured."
    )
    parts.append(
        "\nPlease construct the complete presentation file and output it "
        "using your code execution environment so I can fetch the file_id."
    )
    return "\n".join(parts)


async def create_ppt_db(
    input_data: CreatePptInput,
    sql_db: Session,
    user_id: str,
    conversation_id: str,
    base_url: str,
) -> str:
    """Generate a .pptx file from structured slide data and persist a DB record."""
    storage_path: Optional[str] = None
    queue = asyncio.Queue()
    _active_progress_queues["create_ppt"] = queue

    try:
        # 1. Validate user / org
        user = sql_db.query(User).filter(User.id == user_id).first()
        if not user or not user.org_id:
            return "❌ User or Organization not found."

        # 2. Validate slide count
        if len(input_data.slides) > MAX_SLIDES:
            return f"❌ Too many slides ({len(input_data.slides)}). Maximum is {MAX_SLIDES}."

        # 3. Prepare output path (absolute)
        sanitized_title = re.sub(r'[^\w\s-]', '', input_data.title).strip().replace(' ', '_')
        if not sanitized_title:
            sanitized_title = "presentation"

        unique_suffix = str(uuid.uuid4())[:8]
        filename = f"{sanitized_title}_{unique_suffix}.pptx"

        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        storage_path = str(STORAGE_DIR / filename)
        storage_rel = f"storage/presentations/{filename}"

        # 4. Build Prompt
        prompt = _build_prompt(input_data.title, input_data.slides)

        # 5. Generate via shared Claude abstraction (with progress queue)
        success, claude_text = await generate_document(
            prompt=prompt,
            output_path=storage_path,
            skill_id="pptx",
            file_extension=".pptx",
            progress_queue=queue,
        )
        if not success:
            return f"❌ Failed to generate presentation using Claude.\n\nClaude's output:\n{claude_text}"

        # 6. Check file size
        file_size = os.path.getsize(storage_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            os.remove(storage_path)
            storage_path = None
            return (
                f"❌ Generated file is {file_size / (1024 * 1024):.1f} MB which "
                f"exceeds the 10 MB limit. Please reduce the number of slides."
            )

        # 7. Build public URL (from server config, never from LLM)
        base = base_url.rstrip("/")
        file_url = f"{base}/{storage_rel}"

        # 8. Persist DB record
        record = Presentation(
            conversation_id=conversation_id,
            org_id=user.org_id,
            title=input_data.title,
            file_path=storage_rel,
            file_url=file_url,
        )
        sql_db.add(record)
        sql_db.flush()

        for i, slide_in in enumerate(input_data.slides):
            slide_record = PresentationSlide(
                presentation_id=record.id,
                title=slide_in.title,
                content=slide_in.content,
                slide_type=slide_in.slide_type,
                order=i + 1,
            )
            sql_db.add(slide_record)

        sql_db.commit()
        sql_db.refresh(record)

        tool_response = (
            f"✅ Presentation **'{record.title}'** created successfully!\n"
            f"📊 Total slides provided: {len(input_data.slides)}\n"
            f"📥 **Download link:** {file_url}"
        )
        if claude_text:
            tool_response += f"\n\n---\n### Claude's Design Summary:\n{claude_text}"
        return tool_response

    except Exception as e:
        sql_db.rollback()
        # Clean up orphaned file on failure
        if storage_path and os.path.exists(storage_path):
            try:
                os.remove(storage_path)
                logger.info(f"Cleaned up orphaned file: {storage_path}")
            except OSError:
                pass
        logger.error("Error creating presentation: %s", e, exc_info=True)
        return f"❌ Error creating presentation: {str(e)}"
    finally:
        _active_progress_queues.pop("create_ppt", None)


def get_ppt_link_db(sql_db: Session, conversation_id: str) -> str:
    """Return the download link for the latest presentation in this conversation."""
    try:
        record = (
            sql_db.query(Presentation)
            .filter(Presentation.conversation_id == conversation_id)
            .order_by(Presentation.created_at.desc())
            .first()
        )
        if not record:
            return "📋 No presentation found for this conversation. Use 'create_ppt' to generate one."
        if not record.file_url:
            return "📋 Presentation record found but the file has not been generated yet."
        return (
            f"📥 **Presentation:** '{record.title}'\n"
            f"**Download:** {record.file_url}"
        )
    except Exception as e:
        return f"❌ Error retrieving presentation link: {str(e)}"


# ---------------------------------------------------------------------------
# StructuredTool integration (LangChain)
# ---------------------------------------------------------------------------


def ppt_generation_tool(
    sql_db: Session,
    user_id: str,
    conversation_id: Optional[str] = None,
    base_url: str = "http://localhost:8000",
) -> List[StructuredTool]:
    """Returns LangChain StructuredTools for PowerPoint generation."""
    if not conversation_id:
        logger.warning("ppt_generation_tool called without conversation_id.")
        return []

    # Resolve base_url from settings (never trust LLM input)
    resolved_base_url = getattr(settings, "base_url", base_url)

    async def create_ppt(
        title: str,
        slides: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a professional PowerPoint (.pptx) presentation utilizing Claude's AI skills.

        Provide the title and a list of slide objects containing 'title' and 'content'.
        Returns a download link to the generated document.
        """
        parsed_slides = [
            PptSlideInput(**s) if isinstance(s, dict) else s for s in slides
        ]
        input_obj = CreatePptInput(
            title=title,
            slides=parsed_slides,
        )
        return await create_ppt_db(input_obj, sql_db, user_id, conversation_id, resolved_base_url)

    def get_ppt_link() -> str:
        """Return the download link for the most recently created presentation."""
        return get_ppt_link_db(sql_db, conversation_id)

    class _CreateInput(BaseModel):
        title: str = Field(description="Title of the PowerPoint presentation.")
        slides: List[Dict[str, Any]] = Field(
            description=(
                "List of slide objects. Each must have 'title' (str) and "
                "'content' (str). "
                "Optional 'slide_type': 'bullet', 'text', 'title', 'two_column'. "
                "Max 30 slides."
            )
        )

    return [
        StructuredTool.from_function(
            coroutine=create_ppt,
            name="create_ppt",
            description=(
                "Generate a professional PowerPoint (.pptx) presentation. Pass 'title' and 'slides' "
                "(list of {title, content} objects). Returns a download link. Max 30 slides, 10 MB."
            ),
            args_schema=_CreateInput,
        ),
        StructuredTool.from_function(
            func=get_ppt_link,
            name="get_ppt_link",
            description=(
                "Retrieve the download link for the most recent PowerPoint "
                "created in this conversation."
            ),
            args_schema=None,
        ),
    ]
