"""
Word Document Tool — generates .docx files from structured section data.

Architecture:
  User → Agent → create_word_doc tool
                    ↓
                SectionData (title + content)
                    ↓
                python-docx Document
                    ↓
                storage/word_docs/<uuid>.docx
                    ↓
                Download URL

Resource limits (enforced here, not by the LLM):
  - Max sections   : 50
  - Max file size   : 10 MB
"""

import os
import re
import uuid
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from src.infrastructure.database.models import User, GeneratedWord, WordSection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------

MAX_SECTIONS = 50
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# Input Schemas
# ---------------------------------------------------------------------------


class WordSectionInput(BaseModel):
    """Represents one section in the Word document."""
    title: str = Field(description="Title/heading of this section.")
    content: str = Field(
        description=(
            "Text content for this section. Can be multi-paragraph. "
            "Use newlines to separate paragraphs."
        )
    )
    section_type: str = Field(
        default="text",
        description="Type of section: text, list, table"
    )


class CreateWordInput(BaseModel):
    """Input schema for the create_word_doc tool."""
    title: str = Field(
        description="Title / filename label for the Word document."
    )
    sections: List[WordSectionInput] = Field(
        description=(
            "List of sections for the document. Each section has a 'title' and 'content'. "
            "Example: [{\"title\": \"Executive Summary\", \"content\": \"Revenue is up 5%...\"}]"
        )
    )
    base_url: str = Field(
        description=(
            "Base URL of the backend server (e.g. http://localhost:8000). "
            "Injected automatically — do not ask the user."
        )
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _apply_heading_style(paragraph, level: int = 1) -> None:
    """Apply clean heading formatting."""
    run = paragraph.runs[0] if paragraph.runs else paragraph.add_run("")
    run.bold = True
    if level == 0:  # Document title
        run.font.size = Pt(22)
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif level == 1:  # Section heading
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)
    elif level == 2:  # Sub-heading
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(0x34, 0x49, 0x5E)


async def create_word_db(
    input_data: CreateWordInput,
    sql_db: Session,
    user_id: str,
    conversation_id: str,
) -> str:
    """Generate a .docx file from structured section data and persist a DB record."""
    try:
        # 1. Validate user / org
        user = sql_db.query(User).filter(User.id == user_id).first()
        if not user or not user.org_id:
            return "❌ User or Organization not found."

        # 2. Prepare output path
        sanitized_title = re.sub(r'[^\w\s-]', '', input_data.title).strip().replace(' ', '_')
        if not sanitized_title:
            sanitized_title = "document"

        unique_suffix = str(uuid.uuid4())[:8]
        filename = f"{sanitized_title}_{unique_suffix}.docx"

        storage_rel = f"storage/word_docs/{filename}"
        os.makedirs("storage/word_docs", exist_ok=True)

        # 3. Build Word document with python-docx
        doc = Document()

        # -- Styles: narrower margins for more content space
        for section_obj in doc.sections:
            section_obj.top_margin = Inches(0.8)
            section_obj.bottom_margin = Inches(0.8)
            section_obj.left_margin = Inches(1.0)
            section_obj.right_margin = Inches(1.0)

        # -- Document title
        title_para = doc.add_heading(input_data.title, level=0)
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph("")  # spacer

        # -- Sections
        for sec in input_data.sections:
            doc.add_heading(sec.title, level=1)

            # Split content into paragraphs by double-newline or single-newline
            paragraphs = sec.content.split("\n")
            for para_text in paragraphs:
                para_text = para_text.strip()
                if not para_text:
                    continue

                # Detect bullet points
                if para_text.startswith("- ") or para_text.startswith("* "):
                    p = doc.add_paragraph(para_text[2:], style="List Bullet")
                elif re.match(r'^\d+\.\s+', para_text):
                    text = re.sub(r'^\d+\.\s+', '', para_text)
                    p = doc.add_paragraph(text, style="List Number")
                else:
                    p = doc.add_paragraph(para_text)
                    p.paragraph_format.space_after = Pt(6)

        # 4. Save
        doc.save(storage_rel)

        # 5. Check file size
        file_size = os.path.getsize(storage_rel)
        if file_size > MAX_FILE_SIZE_BYTES:
            os.remove(storage_rel)
            return (
                f"❌ Generated file is {file_size / (1024 * 1024):.1f} MB which "
                f"exceeds the 10 MB limit. Please reduce the content."
            )

        # 6. Build public URL
        base = input_data.base_url.rstrip("/")
        file_url = f"{base}/{storage_rel}"

        # 7. Persist DB record
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

        section_names = ", ".join(f"'{s.title}'" for s in input_data.sections)
        return (
            f"✅ Word document **'{record.title}'** created successfully!\n"
            f"📄 Sections: {section_names}\n"
            f"📥 **Download link:** {file_url}"
        )

    except Exception as e:
        sql_db.rollback()
        logger.error("Error creating Word document: %s", e, exc_info=True)
        return f"❌ Error creating Word document: {str(e)}"


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
    """
    Returns LangChain StructuredTools for Word document generation.
    """
    if not conversation_id:
        logger.warning(
            "word_generation_tool called without conversation_id. "
            "Word tools will not be available."
        )
        return []

    # ------------------------------------------------------------------
    # create_word_doc
    # ------------------------------------------------------------------

    async def create_word_doc(
        title: str,
        sections: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a Word (.docx) document from structured section data and
        return a download link.

        SECTIONS FORMAT — each item must have:
          - "title"   : section heading (str)
          - "content" : section body text (str, can be multi-paragraph)

        EXAMPLE:
          sections = [
            {
              "title": "Executive Summary",
              "content": "Revenue is up 5%. Profits are stable."
            },
            {
              "title": "Market Analysis",
              "content": "The freight rail market is growing..."
            }
          ]

        LIMITS:
          - Max 50 sections
          - Max 10 MB file size
        """
        parsed_sections = [
            WordSectionInput(**s) if isinstance(s, dict) else s for s in sections
        ]
        input_obj = CreateWordInput(
            title=title,
            sections=parsed_sections,
            base_url=base_url,
        )
        return await create_word_db(input_obj, sql_db, user_id, conversation_id)

    # ------------------------------------------------------------------
    # get_word_link
    # ------------------------------------------------------------------

    def get_word_link() -> str:
        """
        Return the download link for the most recently created Word document
        in this conversation.
        """
        return get_word_link_db(sql_db, conversation_id)

    # ------------------------------------------------------------------
    # Register as StructuredTools
    # ------------------------------------------------------------------

    class _CreateInput(BaseModel):
        title: str = Field(description="Title / label for the Word document.")
        sections: List[Dict[str, Any]] = Field(
            description=(
                "List of section objects. Each must have 'title' (str) and "
                "'content' (str). Optional 'section_type' (str). "
                "Max 50 sections. Max 10 MB file size."
            )
        )

    return [
        StructuredTool.from_function(
            coroutine=create_word_doc,
            name="create_word_doc",
            description=(
                "Generate a Word (.docx) document with one or more sections. "
                "Pass 'title' (document name) and 'sections' (list of {title, content} objects). "
                "Returns a download link on success. "
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
