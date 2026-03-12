"""
PowerPoint Tool — generates .pptx files from structured slide data using python-pptx.

Architecture:
  User → Agent → create_ppt tool
                    ↓
                SlideData (title + content + slide_type)
                    ↓
                python-pptx Presentation
                    ↓
                storage/presentations/<name>_<uuid>.pptx
                    ↓
                Download URL

Design System:
  - Strict color palette (dark navy / slate / accent blue)
  - Consistent typography hierarchy (title → subtitle → body)
  - Uniform spacing & margins across all slides
  - Professional slide layouts: title slide, bullet slides, text slides

Resource limits:
  - Max slides     : 30
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

from pptx import Presentation as PptxPresentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

from src.infrastructure.database.models import User, Presentation, PresentationSlide

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design System Constants
# ---------------------------------------------------------------------------

# Color palette
COLOR_BG_DARK     = RGBColor(0x1A, 0x1A, 0x2E)   # Deep navy background
COLOR_BG_SLIDE    = RGBColor(0xFF, 0xFF, 0xFF)     # White slide background
COLOR_TITLE       = RGBColor(0x1A, 0x1A, 0x2E)     # Dark navy for titles
COLOR_SUBTITLE    = RGBColor(0x4A, 0x4A, 0x6A)     # Slate for subtitles
COLOR_BODY        = RGBColor(0x33, 0x33, 0x44)     # Dark gray for body text
COLOR_ACCENT      = RGBColor(0x2D, 0x7D, 0xD2)     # Accent blue for highlights
COLOR_BULLET      = RGBColor(0x2D, 0x7D, 0xD2)     # Accent blue for bullet markers
COLOR_DIVIDER     = RGBColor(0xE0, 0xE4, 0xE8)     # Light gray divider
COLOR_WHITE       = RGBColor(0xFF, 0xFF, 0xFF)

# Typography
FONT_FAMILY       = "Calibri"
FONT_TITLE_SIZE   = Pt(32)
FONT_HEADING_SIZE = Pt(24)
FONT_BODY_SIZE    = Pt(16)
FONT_SMALL_SIZE   = Pt(12)

# Layout
SLIDE_WIDTH       = Inches(13.333)  # 16:9 widescreen
SLIDE_HEIGHT      = Inches(7.5)
MARGIN_LEFT       = Inches(0.8)
MARGIN_TOP        = Inches(0.6)
CONTENT_WIDTH     = Inches(11.7)
CONTENT_TOP       = Inches(1.6)
CONTENT_HEIGHT    = Inches(5.2)

# Limits
MAX_SLIDES = 30
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# ---------------------------------------------------------------------------
# Input Schemas
# ---------------------------------------------------------------------------


class PptSlideInput(BaseModel):
    """Represents one slide in the presentation."""
    title: str = Field(description="Title for this slide.")
    content: str = Field(
        description=(
            "Content for this slide. Use newlines to separate bullet points. "
            "Prefix lines with '- ' for bullet points."
        )
    )
    slide_type: str = Field(
        default="bullet",
        description="Type of slide: 'bullet' for bullet-point slides, 'text' for paragraph slides, 'title' for a title/section divider slide."
    )


class CreatePptInput(BaseModel):
    """Input schema for the create_ppt tool."""
    title: str = Field(description="Title of the PowerPoint presentation.")
    slides: List[PptSlideInput] = Field(
        description=(
            "List of slides. Each has 'title', 'content', and optional 'slide_type'. "
            "Example: [{\"title\": \"Overview\", \"content\": \"- Point 1\\n- Point 2\", \"slide_type\": \"bullet\"}]"
        )
    )
    base_url: str = Field(
        description="Base URL of the backend server. Injected automatically."
    )


# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------


def _set_slide_bg(slide, color: RGBColor) -> None:
    """Set the background color of a slide."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_shape_with_text(slide, left, top, width, height, text: str,
                         font_size=FONT_BODY_SIZE, font_color=COLOR_BODY,
                         bold=False, alignment=PP_ALIGN.LEFT) -> None:
    """Add a text box with consistent styling."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    # Split by newline and handle bullet points
    lines = text.split("\n")
    first = True
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()

        is_bullet = line.startswith("- ") or line.startswith("* ")
        if is_bullet:
            line = line[2:]
            p.level = 0
            # Add a bullet marker manually via text
            run = p.add_run()
            run.text = "●  "
            run.font.size = Pt(10)
            run.font.color.rgb = COLOR_BULLET
            run.font.name = FONT_FAMILY

        run = p.add_run()
        run.text = line
        run.font.size = font_size
        run.font.color.rgb = font_color
        run.font.name = FONT_FAMILY
        run.font.bold = bold

        p.alignment = alignment
        p.space_after = Pt(8)
        p.space_before = Pt(2)


def _build_title_slide(prs, title: str, subtitle: str = "") -> None:
    """Build a cover/title slide with dark background."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
    _set_slide_bg(slide, COLOR_BG_DARK)

    # Title text
    _add_shape_with_text(
        slide,
        left=MARGIN_LEFT, top=Inches(2.5),
        width=CONTENT_WIDTH, height=Inches(1.2),
        text=title,
        font_size=Pt(40), font_color=COLOR_WHITE,
        bold=True, alignment=PP_ALIGN.CENTER
    )

    # Subtitle / description
    if subtitle:
        _add_shape_with_text(
            slide,
            left=MARGIN_LEFT, top=Inches(4.0),
            width=CONTENT_WIDTH, height=Inches(0.8),
            text=subtitle,
            font_size=Pt(18), font_color=RGBColor(0xAA, 0xBB, 0xCC),
            alignment=PP_ALIGN.CENTER
        )

    # Accent divider line
    from pptx.util import Emu as EmuUtil
    shape = slide.shapes.add_shape(
        1,  # Rectangle
        Inches(5.0), Inches(3.9),
        Inches(3.3), Pt(3)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = COLOR_ACCENT
    shape.line.fill.background()


def _build_content_slide(prs, title: str, content: str, slide_type: str = "bullet") -> None:
    """Build a content slide with white background and consistent hierarchy."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
    _set_slide_bg(slide, COLOR_BG_SLIDE)

    # Slide title
    _add_shape_with_text(
        slide,
        left=MARGIN_LEFT, top=MARGIN_TOP,
        width=CONTENT_WIDTH, height=Inches(0.7),
        text=title,
        font_size=FONT_HEADING_SIZE, font_color=COLOR_TITLE,
        bold=True
    )

    # Accent divider under title
    shape = slide.shapes.add_shape(
        1,  # Rectangle
        MARGIN_LEFT, Inches(1.3),
        Inches(2.0), Pt(3)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = COLOR_ACCENT
    shape.line.fill.background()

    # Content body
    _add_shape_with_text(
        slide,
        left=MARGIN_LEFT, top=CONTENT_TOP,
        width=CONTENT_WIDTH, height=CONTENT_HEIGHT,
        text=content,
        font_size=FONT_BODY_SIZE, font_color=COLOR_BODY,
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


async def create_ppt_db(
    input_data: CreatePptInput,
    sql_db: Session,
    user_id: str,
    conversation_id: str,
) -> str:
    """Generate a .pptx file from structured slide data and persist a DB record."""
    try:
        # 1. Validate user / org
        user = sql_db.query(User).filter(User.id == user_id).first()
        if not user or not user.org_id:
            return "❌ User or Organization not found."

        # 2. Prepare output path
        sanitized_title = re.sub(r'[^\w\s-]', '', input_data.title).strip().replace(' ', '_')
        if not sanitized_title:
            sanitized_title = "presentation"

        unique_suffix = str(uuid.uuid4())[:8]
        filename = f"{sanitized_title}_{unique_suffix}.pptx"

        storage_rel = f"storage/presentations/{filename}"
        os.makedirs("storage/presentations", exist_ok=True)

        # 3. Build PowerPoint with python-pptx
        prs = PptxPresentation()
        prs.slide_width = SLIDE_WIDTH
        prs.slide_height = SLIDE_HEIGHT

        # Title slide (first slide is always the cover)
        first_slide_content = input_data.slides[0].content if input_data.slides else ""
        _build_title_slide(prs, input_data.title, first_slide_content if input_data.slides and input_data.slides[0].slide_type == "title" else "")

        # Content slides
        for slide_data in input_data.slides:
            if slide_data.slide_type == "title":
                # If it's a section divider, use the title slide style
                _build_title_slide(prs, slide_data.title, slide_data.content)
            else:
                _build_content_slide(prs, slide_data.title, slide_data.content, slide_data.slide_type)

        # 4. Save
        prs.save(storage_rel)

        # 5. Check file size
        file_size = os.path.getsize(storage_rel)
        if file_size > MAX_FILE_SIZE_BYTES:
            os.remove(storage_rel)
            return (
                f"❌ Generated file is {file_size / (1024 * 1024):.1f} MB which "
                f"exceeds the 10 MB limit. Please reduce the number of slides."
            )

        # 6. Build public URL
        base = input_data.base_url.rstrip("/")
        file_url = f"{base}/{storage_rel}"

        # 7. Persist DB record
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

        total_slides = len(input_data.slides) + 1  # +1 for cover
        return (
            f"✅ Presentation **'{record.title}'** created successfully!\n"
            f"📊 Total slides: {total_slides} (including cover)\n"
            f"📥 **Download link:** {file_url}"
        )

    except Exception as e:
        sql_db.rollback()
        logger.error("Error creating presentation: %s", e, exc_info=True)
        return f"❌ Error creating presentation: {str(e)}"


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
    """
    Returns LangChain StructuredTools for PowerPoint generation.
    """
    if not conversation_id:
        logger.warning(
            "ppt_generation_tool called without conversation_id. "
            "PPT tools will not be available."
        )
        return []

    async def create_ppt(
        title: str,
        slides: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a PowerPoint (.pptx) presentation from structured slide data
        and return a download link.

        SLIDES FORMAT — each item must have:
          - "title"      : slide heading (str)
          - "content"    : slide body text (str). Use '- ' prefix for bullet points.
          - "slide_type" : (optional) "bullet" (default), "text", or "title" (section divider)

        EXAMPLE:
          slides = [
            {
              "title": "Market Overview",
              "content": "- Revenue grew 15% YoY\\n- Market share expanded to 23%\\n- Customer base doubled",
              "slide_type": "bullet"
            },
            {
              "title": "Strategic Priorities",
              "content": "Focus on three core pillars:\\n- Innovation pipeline\\n- Customer retention\\n- Operational efficiency",
              "slide_type": "bullet"
            }
          ]

        LIMITS:
          - Max 30 slides
          - Max 10 MB file size
        """
        parsed_slides = [
            PptSlideInput(**s) if isinstance(s, dict) else s for s in slides
        ]
        input_obj = CreatePptInput(
            title=title,
            slides=parsed_slides,
            base_url=base_url,
        )
        return await create_ppt_db(input_obj, sql_db, user_id, conversation_id)

    def get_ppt_link() -> str:
        """
        Return the download link for the most recently created presentation
        in this conversation.
        """
        return get_ppt_link_db(sql_db, conversation_id)

    class _CreateInput(BaseModel):
        title: str = Field(description="Title of the PowerPoint presentation.")
        slides: List[Dict[str, Any]] = Field(
            description=(
                "List of slide objects. Each must have 'title' (str) and "
                "'content' (str, use '- ' for bullets). Optional 'slide_type' "
                "(str: 'bullet', 'text', or 'title'). Max 30 slides."
            )
        )

    return [
        StructuredTool.from_function(
            coroutine=create_ppt,
            name="create_ppt",
            description=(
                "Generate a PowerPoint (.pptx) presentation with professional styling. "
                "Pass 'title' (presentation name) and 'slides' (list of {title, content} objects). "
                "Use '- ' prefix in content for bullet points. "
                "Set slide_type to 'title' for section dividers. "
                "Returns a download link on success. Max 30 slides, 10 MB."
            ),
            args_schema=_CreateInput,
        ),
        StructuredTool.from_function(
            func=get_ppt_link,
            name="get_ppt_link",
            description=(
                "Retrieve the download link for the most recent PowerPoint presentation "
                "created in this conversation."
            ),
            args_schema=None,
        ),
    ]
