"""
Word Document Tool — generates professional .docx files using Claude's code execution skill.

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

from src.infrastructure.database.models import User, GeneratedWord, WordSection
from src.infrastructure.agents.tools.claude_document_generator import (
    generate_document,
    _QUEUE_DONE,
)
from src.config.settings import settings

logger = logging.getLogger(__name__)

MAX_SECTIONS = 50
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Absolute storage root (derived from project base, not CWD)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # …/backend
STORAGE_DIR = _PROJECT_ROOT / "storage" / "word_docs"

# Global registry for active progress queues
_active_progress_queues: Dict[str, asyncio.Queue] = {}


def get_active_progress_queue(tool_name: str) -> Optional[asyncio.Queue]:
    """Get the active progress queue for a running tool, if any."""
    return _active_progress_queues.get(tool_name)


class WordSectionInput(BaseModel):
    """Represents one section in the Word document."""
    title: str = Field(description="Title/heading of this section.")
    content: str = Field(
        description="Text content for this section. Supports formatting."
    )
    section_type: str = Field(
        default="text",
        description="Type of section: text, list, table",
    )


class CreateWordInput(BaseModel):
    """Input schema for the create_word_doc tool."""
    title: str = Field(
        description="Title / filename label for the Word document."
    )
    sections: List[WordSectionInput] = Field(
        description="List of sections for the document."
    )


def _build_prompt(title: str, sections: List[WordSectionInput]) -> str:
    """Build the Claude prompt from structured section data."""
    parts = [
        f"Generate a highly professional Word document (.docx) titled '{title}'.",
        "Use Python's python-docx library to generate the document organically, save it, and ensure we get the file down.\n",
        "Document Sections:",
    ]
    for idx, sec in enumerate(sections):
        parts.append(f"--- Section {idx + 1} ---")
        parts.append(f"Type: {sec.section_type}")
        parts.append(f"Title: {sec.title}")
        parts.append(f"Content:\n{sec.content}\n")

    parts.append(
        "\nIMPORTANT: Please generate only the .docx document file. "
        "Avoid creating summary text files or extra reports in the environment "
        "to ensure the correct file is captured."
    )
    parts.append(
        "\nPlease construct the complete Word document and output it "
        "using your code execution environment so I can fetch the file_id."
    )
    return "\n".join(parts)


async def create_word_db(
    input_data: CreateWordInput,
    sql_db: Session,
    user_id: str,
    conversation_id: str,
    base_url: str,
) -> str:
    """Generate a .docx file from structured section data and persist a DB record."""
    storage_path: Optional[str] = None
    queue = asyncio.Queue()
    _active_progress_queues["create_word_doc"] = queue

    try:
        # 1. Validate user / org
        user = sql_db.query(User).filter(User.id == user_id).first()
        if not user or not user.org_id:
            return "❌ User or Organization not found."

        # 2. Validate section count
        if len(input_data.sections) > MAX_SECTIONS:
            return f"❌ Too many sections ({len(input_data.sections)}). Maximum is {MAX_SECTIONS}."

        # 3. Prepare output path (absolute)
        sanitized_title = re.sub(r'[^\w\s-]', '', input_data.title).strip().replace(' ', '_')
        if not sanitized_title:
            sanitized_title = "document"

        unique_suffix = str(uuid.uuid4())[:8]
        filename = f"{sanitized_title}_{unique_suffix}.docx"

        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        storage_path = str(STORAGE_DIR / filename)
        storage_rel = f"storage/word_docs/{filename}"

        # 4. Build Prompt
        prompt = _build_prompt(input_data.title, input_data.sections)

        # 5. Generate via shared Claude abstraction (with progress queue)
        success, claude_text = await generate_document(
            prompt=prompt,
            output_path=storage_path,
            skill_id="docx",
            file_extension=".docx",
            progress_queue=queue,
        )
        if not success:
            return f"❌ Failed to generate document using Claude.\n\nClaude's output:\n{claude_text}"

        # 6. Check file size
        file_size = os.path.getsize(storage_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            os.remove(storage_path)
            storage_path = None
            return (
                f"❌ Generated file is {file_size / (1024 * 1024):.1f} MB which "
                f"exceeds the 10 MB limit. Please reduce the content."
            )

        # 7. Build public URL (from server config, never from LLM)
        base = base_url.rstrip("/")
        file_url = f"{base}/{storage_rel}"

        # 8. Persist DB record
        record = GeneratedWord(
            conversation_id=conversation_id,
            org_id=user.org_id,
            title=input_data.title,
            file_path=storage_rel,
            file_url=file_url,
        )
        sql_db.add(record)
        sql_db.flush()

        for i, sec_in in enumerate(input_data.sections):
            section_record = WordSection(
                generated_word_id=record.id,
                title=sec_in.title,
                content=sec_in.content,
                section_type=sec_in.section_type,
                order=i + 1,
            )
            sql_db.add(section_record)

        sql_db.commit()
        sql_db.refresh(record)

        tool_response = (
            f"✅ Word document **'{record.title}'** created successfully!\n"
            f"📄 Total sections provided: {len(input_data.sections)}\n"
            f"📥 **Download link:** {file_url}"
        )
        if claude_text:
            tool_response += f"\n\n---\n### Claude's Document Generation Summary:\n{claude_text}"
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
        logger.error("Error creating Word document: %s", e, exc_info=True)
        return f"❌ Error creating Word document: {str(e)}"
    finally:
        _active_progress_queues.pop("create_word_doc", None)


def get_word_link_db(sql_db: Session, conversation_id: str) -> str:
    """Return the download link for the latest Word document in this conversation."""
    try:
        record = (
            sql_db.query(GeneratedWord)
            .filter(GeneratedWord.conversation_id == conversation_id)
            .order_by(GeneratedWord.created_at.desc())
            .first()
        )
        if not record:
            return "📋 No Word document found for this conversation. Use 'create_word_doc' to generate one."
        if not record.file_url:
            return "📋 Word document record found but the file has not been generated yet."
        return (
            f"📥 **Word Document:** '{record.title}'\n"
            f"**Download:** {record.file_url}"
        )
    except Exception as e:
        return f"❌ Error retrieving Word document link: {str(e)}"


# ---------------------------------------------------------------------------
# StructuredTool integration (LangChain)
# ---------------------------------------------------------------------------


def word_generation_tool(
    sql_db: Session,
    user_id: str,
    conversation_id: Optional[str] = None,
    base_url: str = "http://localhost:8000",
) -> List[StructuredTool]:
    """Returns LangChain StructuredTools for Word document generation."""
    if not conversation_id:
        logger.warning(
            "word_generation_tool called without conversation_id. "
            "Word tools will not be available."
        )
        return []

    # Resolve base_url from settings (never trust LLM input)
    resolved_base_url = getattr(settings, "base_url", base_url)

    async def create_word_doc(
        title: str,
        sections: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a professional Word (.docx) document utilizing Claude's AI skills.

        Provide the title and a list of section objects containing 'title' and 'content'.
        Returns a download link to the generated document.
        """
        parsed_sections = [
            WordSectionInput(**s) if isinstance(s, dict) else s for s in sections
        ]
        input_obj = CreateWordInput(
            title=title,
            sections=parsed_sections,
        )
        return await create_word_db(input_obj, sql_db, user_id, conversation_id, resolved_base_url)

    def get_word_link() -> str:
        """Return the download link for the most recently created Word document."""
        return get_word_link_db(sql_db, conversation_id)

    class _CreateInput(BaseModel):
        title: str = Field(description="Title / label for the Word document.")
        sections: List[Dict[str, Any]] = Field(
            description=(
                "List of section objects. Each must have 'title' (str) and "
                "'content' (str). Max 50 sections. Max 10 MB file size."
            )
        )

    return [
        StructuredTool.from_function(
            coroutine=create_word_doc,
            name="create_word_doc",
            description=(
                "Generate a professional Word (.docx) document. Pass 'title' (document name) and "
                "'sections' (list of {title, content} objects). Returns a download link on success. "
                "Limits: max 50 sections, 10 MB."
            ),
            args_schema=_CreateInput,
        ),
        StructuredTool.from_function(
            func=get_word_link,
            name="get_word_link",
            description=(
                "Retrieve the download link for the most recent Word document "
                "created in this conversation."
            ),
            args_schema=None,
        ),
    ]
