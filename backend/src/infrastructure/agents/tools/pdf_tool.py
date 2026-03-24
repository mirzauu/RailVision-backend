"""
PDF Document Tool — generates professional PDF files from structured section data.

Architecture:
  User → Agent → create_pdf tool
                    ↓
                SectionData (title + content with markdown)
                    ↓
                ReportLab PDF Document (with rich formatting + charts)
                    ↓
                storage/pdfs/<uuid>.pdf
                    ↓
                Download URL

Supported Markdown in content:
  - ## Sub-Heading / ### Sub-Sub-Heading
  - **bold**, *italic*
  - | col1 | col2 | table syntax
  - - bullet / * bullet
  - 1. numbered items
  - > blockquote
  - --- horizontal rule
  - ```chart ... ``` blocks (bar, line, pie)

Resource limits (enforced here, not by the LLM):
  - Max sections   : 50
  - Max file size   : 10 MB
"""

import os
import re
import io
import uuid
import math
import logging
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool
from fpdf import FPDF

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from src.infrastructure.database.models import User, GeneratedPDF, PDFSection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------

MAX_SECTIONS = 50
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# Color Palette — Professional dark-blue theme (RGB tuples for fpdf)
# ---------------------------------------------------------------------------

COLORS = {
    "primary":       (0x1B, 0x2A, 0x4A),   # Dark navy
    "secondary":     (0x2C, 0x5F, 0x8A),   # Steel blue
    "accent":        (0x3A, 0x86, 0xC8),   # Bright blue
    "heading_1":     (0x1B, 0x2A, 0x4A),   # Navy
    "heading_2":     (0x2C, 0x5F, 0x8A),   # Steel blue
    "heading_3":     (0x3A, 0x86, 0xC8),   # Bright blue
    "body":          (0x2D, 0x2D, 0x2D),   # Dark gray
    "light_gray":    (0x6B, 0x70, 0x7B),   # Muted gray
    "table_header":  (0x1B, 0x2A, 0x4A),   # Navy
    "table_alt_row": (0xF0, 0xF4, 0xF8),   # Light blue-gray
    "border":        (0xD0, 0xD5, 0xDD),   # Border gray
    "blockquote":    (0x4A, 0x5A, 0x6A),   # Quote gray
    "white":         (0xFF, 0xFF, 0xFF),
    "cover_line":    (0x3A, 0x86, 0xC8),   # Accent for cover
}

# Chart color palette
CHART_COLORS = [
    (0x3A, 0x86, 0xC8),   # Blue
    (0xE8, 0x6C, 0x50),   # Coral-red
    (0x2E, 0xCC, 0x71),   # Green
    (0xF3, 0x9C, 0x12),   # Amber
    (0x9B, 0x59, 0xB6),   # Purple
    (0x1A, 0xBC, 0x9C),   # Teal
    (0xE7, 0x4C, 0x3C),   # Red
    (0x34, 0x98, 0xDB),   # Light blue
    (0xF1, 0xC4, 0x0F),   # Yellow
    (0x2C, 0x3E, 0x50),   # Dark slate
]


# ---------------------------------------------------------------------------
# Input Schemas
# ---------------------------------------------------------------------------


class PDFSectionInput(BaseModel):
    """Represents one section in the PDF document."""
    title: str = Field(description="Title/heading of this section.")
    content: str = Field(
        description=(
            "Text content for this section. Supports markdown-like formatting: "
            "**bold**, *italic*, ## Sub-Heading, ### Sub-Sub-Heading, "
            "- bullet points, 1. numbered items, > blockquotes, "
            "--- horizontal rules, | col1 | col2 | table syntax, "
            "and ```chart blocks."
        )
    )
    section_type: str = Field(
        default="text",
        description="Type of section: text, list, table"
    )


class CreatePDFInput(BaseModel):
    """Input schema for the create_pdf tool."""
    title: str = Field(description="Title of the PDF report")
    sections: List[PDFSectionInput] = Field(
        description="List of sections for the report. Content supports markdown formatting."
    )
    base_url: str = Field(description="Base URL of the backend server")


# ---------------------------------------------------------------------------
# Text sanitization for fpdf (latin-1 safe)
# ---------------------------------------------------------------------------


def _sanitize_text(text: str) -> str:
    """Sanitize text for fpdf2 latin-1 encoding."""
    if not text:
        return ""
    replacements = {
        '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
        '\u2014': '-', '\u2013': '-', '\u2026': '...',
        '\u2022': '-', '\u2023': '-', '\u25e6': '-',
        '\u2713': '[x]', '\u2717': '[ ]',
        '\u2192': '->', '\u2190': '<-',
        '\u2500': '-', '\u2501': '-', '\u2550': '=',
        '\u2588': '#', '\u2591': '.',
        '\u25cf': '*', '\u25cb': 'o',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')


# ---------------------------------------------------------------------------
# Professional PDF class with header/footer
# ---------------------------------------------------------------------------


class ProfessionalPDF(FPDF):
    """Custom PDF with professional header, footer, and utility methods."""

    def __init__(self, doc_title: str = "Document"):
        super().__init__()
        self.doc_title = doc_title
        self._is_cover_page = True
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self._is_cover_page:
            return
        # Right-aligned document title in header
        self.set_font("helvetica", "I", 8)
        self.set_text_color(*COLORS["light_gray"])
        self.cell(0, 8, _sanitize_text(self.doc_title), align="R")
        # Subtle line under header
        self.set_draw_color(*COLORS["border"])
        self.line(15, 14, self.w - 15, 14)
        self.ln(12)

    def footer(self):
        if self._is_cover_page:
            return
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(*COLORS["light_gray"])
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------


def _add_cover_page(pdf: ProfessionalPDF, title: str) -> None:
    """Add a professional cover/title page."""
    pdf._is_cover_page = True
    pdf.add_page()

    # Push content to upper-third
    pdf.ln(60)

    # Accent line
    pdf.set_draw_color(*COLORS["cover_line"])
    pdf.set_line_width(1.5)
    center_x = pdf.w / 2
    pdf.line(center_x - 60, pdf.get_y(), center_x + 60, pdf.get_y())
    pdf.ln(12)

    # Title
    pdf.set_font("helvetica", "B", 28)
    pdf.set_text_color(*COLORS["primary"])
    pdf.multi_cell(0, 14, _sanitize_text(title), align="C")
    pdf.ln(8)

    # Accent line below title
    y = pdf.get_y()
    pdf.line(center_x - 60, y, center_x + 60, y)
    pdf.ln(20)

    # Date
    pdf.set_font("helvetica", "", 12)
    pdf.set_text_color(*COLORS["light_gray"])
    pdf.cell(0, 8, datetime.now().strftime("%B %d, %Y"), align="C")
    pdf.ln(40)

    # Confidential notice
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(*COLORS["light_gray"])
    pdf.cell(0, 8, "CONFIDENTIAL", align="C")

    pdf.set_line_width(0.2)  # Reset line width


# ---------------------------------------------------------------------------
# Table of Contents
# ---------------------------------------------------------------------------


def _add_table_of_contents(pdf: ProfessionalPDF, sections: List[PDFSectionInput]) -> None:
    """Add a table of contents page."""
    pdf._is_cover_page = False
    pdf.add_page()

    # TOC heading
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(*COLORS["heading_1"])
    pdf.cell(0, 12, "Table of Contents", ln=1)
    pdf.ln(8)

    for i, sec in enumerate(sections, 1):
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(*COLORS["accent"])
        num_text = f"{i}.  "
        num_w = pdf.get_string_width(num_text) + 2
        pdf.cell(num_w, 8, num_text)

        pdf.set_font("helvetica", "", 11)
        pdf.set_text_color(*COLORS["body"])
        pdf.cell(0, 8, _sanitize_text(sec.title), ln=1)
        pdf.ln(2)


# ---------------------------------------------------------------------------
# Chart rendering (using Pillow)
# ---------------------------------------------------------------------------


def _get_text_size(draw, text: str, font) -> Tuple[int, int]:
    """Get text bounding box size."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def _get_font(size: int, bold: bool = False):
    """Try to load a good font, fall back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _render_bar_chart(title: str, labels: List[str], values: List[float], width: int = 800, height: int = 500) -> Image.Image:
    """Render a bar chart as a PIL Image."""
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = _get_font(18, bold=True)
    label_font = _get_font(12)
    value_font = _get_font(11, bold=True)

    tw, th = _get_text_size(draw, title, title_font)
    draw.text(((width - tw) // 2, 15), title, fill=COLORS["primary"], font=title_font)

    margin_left, margin_right, margin_top, margin_bottom = 80, 40, 60, 80
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    if not values:
        return img

    max_val = max(values) if max(values) > 0 else 1
    n = len(labels)
    bar_width = max(20, min(60, chart_w // (n * 2)))
    spacing = (chart_w - bar_width * n) / (n + 1)

    # Y-axis gridlines
    for i in range(6):
        y = margin_top + chart_h - (i * chart_h / 5)
        val = (i * max_val / 5)
        draw.line([(margin_left, int(y)), (width - margin_right, int(y))], fill=(0xE0, 0xE0, 0xE0), width=1)
        val_str = f"{val:,.0f}" if val == int(val) else f"{val:,.1f}"
        vw, vh = _get_text_size(draw, val_str, label_font)
        draw.text((margin_left - vw - 8, int(y) - vh // 2), val_str, fill=COLORS["light_gray"], font=label_font)

    # Bars
    for i, (label, val) in enumerate(zip(labels, values)):
        x = margin_left + spacing + i * (bar_width + spacing)
        bar_h = (val / max_val) * chart_h if max_val > 0 else 0
        y_top = margin_top + chart_h - bar_h
        y_bottom = margin_top + chart_h
        color = CHART_COLORS[i % len(CHART_COLORS)]
        draw.rectangle([int(x), int(y_top), int(x + bar_width), int(y_bottom)], fill=color)

        val_str = f"{val:,.0f}" if val == int(val) else f"{val:,.1f}"
        vw, vh = _get_text_size(draw, val_str, value_font)
        draw.text((int(x + bar_width / 2 - vw / 2), int(y_top - vh - 4)), val_str, fill=color, font=value_font)

        lw, lh = _get_text_size(draw, label, label_font)
        draw.text((int(x + bar_width / 2 - lw / 2), int(y_bottom + 8)), label, fill=COLORS["body"], font=label_font)

    draw.line([(margin_left, margin_top), (margin_left, margin_top + chart_h)], fill=COLORS["body"], width=2)
    draw.line([(margin_left, margin_top + chart_h), (width - margin_right, margin_top + chart_h)], fill=COLORS["body"], width=2)
    return img


def _render_line_chart(title: str, labels: List[str], values: List[float], width: int = 800, height: int = 500) -> Image.Image:
    """Render a line chart as a PIL Image."""
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = _get_font(18, bold=True)
    label_font = _get_font(12)
    value_font = _get_font(10, bold=True)

    tw, th = _get_text_size(draw, title, title_font)
    draw.text(((width - tw) // 2, 15), title, fill=COLORS["primary"], font=title_font)

    margin_left, margin_right, margin_top, margin_bottom = 80, 40, 60, 80
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    if not values or len(values) < 2:
        return img

    max_val = max(values) if max(values) > 0 else 1
    min_val = min(values)
    val_range = max_val - min_val if max_val != min_val else 1
    n = len(labels)

    for i in range(6):
        y = margin_top + chart_h - (i * chart_h / 5)
        val = min_val + (i * val_range / 5)
        draw.line([(margin_left, int(y)), (width - margin_right, int(y))], fill=(0xE0, 0xE0, 0xE0), width=1)
        val_str = f"{val:,.0f}" if val == int(val) else f"{val:,.1f}"
        vw, vh = _get_text_size(draw, val_str, label_font)
        draw.text((margin_left - vw - 8, int(y) - vh // 2), val_str, fill=COLORS["light_gray"], font=label_font)

    points = []
    step = chart_w / (n - 1) if n > 1 else chart_w
    for i, val in enumerate(values):
        x = margin_left + i * step
        y = margin_top + chart_h - ((val - min_val) / val_range) * chart_h
        points.append((int(x), int(y)))

    color = CHART_COLORS[0]

    # Area fill
    overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    area_points = points + [(points[-1][0], margin_top + chart_h), (points[0][0], margin_top + chart_h)]
    overlay_draw.polygon(area_points, fill=(color[0], color[1], color[2], 40))
    img.paste(Image.alpha_composite(Image.new('RGBA', img.size, (255, 255, 255, 255)), overlay).convert('RGB'))
    draw = ImageDraw.Draw(img)

    # Re-draw gridlines
    for i in range(6):
        y = margin_top + chart_h - (i * chart_h / 5)
        draw.line([(margin_left, int(y)), (width - margin_right, int(y))], fill=(0xE0, 0xE0, 0xE0), width=1)

    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=color, width=3)

    for i, (px, py) in enumerate(points):
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=color, outline=(255, 255, 255))
        val_str = f"{values[i]:,.0f}" if values[i] == int(values[i]) else f"{values[i]:,.1f}"
        vw, vh = _get_text_size(draw, val_str, value_font)
        draw.text((px - vw // 2, py - vh - 8), val_str, fill=color, font=value_font)

    for i, label in enumerate(labels):
        x = margin_left + i * step
        lw, lh = _get_text_size(draw, label, label_font)
        draw.text((int(x - lw / 2), margin_top + chart_h + 8), label, fill=COLORS["body"], font=label_font)

    draw.line([(margin_left, margin_top), (margin_left, margin_top + chart_h)], fill=COLORS["body"], width=2)
    draw.line([(margin_left, margin_top + chart_h), (width - margin_right, margin_top + chart_h)], fill=COLORS["body"], width=2)
    return img


def _render_pie_chart(title: str, labels: List[str], values: List[float], width: int = 700, height: int = 500) -> Image.Image:
    """Render a pie chart as a PIL Image."""
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = _get_font(18, bold=True)
    label_font = _get_font(12)

    tw, th = _get_text_size(draw, title, title_font)
    draw.text(((width - tw) // 2, 15), title, fill=COLORS["primary"], font=title_font)

    if not values or sum(values) == 0:
        return img

    total = sum(values)
    center_x = width // 2 - 60
    center_y = height // 2 + 15
    radius = min(width, height) // 2 - 80

    start_angle = -90
    for i, (label, val) in enumerate(zip(labels, values)):
        sweep = (val / total) * 360
        color = CHART_COLORS[i % len(CHART_COLORS)]
        bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        draw.pieslice(bbox, start_angle, start_angle + sweep, fill=color, outline=(255, 255, 255), width=2)
        start_angle += sweep

    legend_x = center_x + radius + 40
    legend_y = center_y - (len(labels) * 25) // 2
    for i, (label, val) in enumerate(zip(labels, values)):
        color = CHART_COLORS[i % len(CHART_COLORS)]
        y = legend_y + i * 28
        draw.rectangle([legend_x, y, legend_x + 16, y + 16], fill=color)
        pct = (val / total) * 100
        draw.text((legend_x + 24, y), f"{label} ({pct:.1f}%)", fill=COLORS["body"], font=label_font)

    return img


def _parse_chart_block(chart_text: str) -> Optional[Dict[str, Any]]:
    """Parse a chart definition block."""
    result = {"type": "bar", "title": "Chart", "labels": [], "values": []}
    lines = chart_text.strip().split("\n")
    in_data = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("type:"):
            result["type"] = stripped.split(":", 1)[1].strip().lower()
            in_data = False
        elif stripped.lower().startswith("title:"):
            result["title"] = stripped.split(":", 1)[1].strip()
            in_data = False
        elif stripped.lower() == "data:":
            in_data = True
        elif in_data and ":" in stripped:
            parts = stripped.rsplit(":", 1)
            if len(parts) == 2:
                label = parts[0].strip()
                try:
                    value = float(parts[1].strip().replace(",", "").replace("$", "").replace("%", ""))
                    result["labels"].append(label)
                    result["values"].append(value)
                except ValueError:
                    continue

    if not result["labels"]:
        return None
    return result


def _add_chart_to_pdf(pdf: ProfessionalPDF, chart_data: Dict[str, Any]) -> None:
    """Render a chart and add it as an image to the PDF."""
    if not PIL_AVAILABLE:
        pdf.set_font("helvetica", "I", 10)
        pdf.set_text_color(*COLORS["light_gray"])
        pdf.cell(0, 8, f"[Chart: {chart_data.get('title', 'Chart')} - install Pillow for visual charts]", ln=1)
        return

    chart_type = chart_data.get("type", "bar")
    title = chart_data.get("title", "Chart")
    labels = chart_data.get("labels", [])
    values = chart_data.get("values", [])

    try:
        if chart_type == "line":
            chart_img = _render_line_chart(title, labels, values)
        elif chart_type == "pie":
            chart_img = _render_pie_chart(title, labels, values)
        else:
            chart_img = _render_bar_chart(title, labels, values)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            chart_img.save(tmp, format="PNG", quality=95)
            tmp_path = tmp.name

        # Center the chart image
        img_width = 160  # mm
        x = (pdf.w - img_width) / 2
        pdf.image(tmp_path, x=x, w=img_width)
        pdf.ln(8)

        try:
            os.remove(tmp_path)
        except OSError:
            pass

    except Exception as e:
        logger.warning(f"Failed to render chart in PDF: {e}")
        pdf.set_font("helvetica", "I", 10)
        pdf.set_text_color(*COLORS["light_gray"])
        pdf.cell(0, 8, f"[Chart rendering failed: {str(e)}]", ln=1)


# ---------------------------------------------------------------------------
# Content rendering — markdown parsing → PDF formatting
# ---------------------------------------------------------------------------


def _render_table(pdf: ProfessionalPDF, lines: List[str]) -> None:
    """Parse and render a markdown table in the PDF."""
    rows_data = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if re.match(r'^\|[\s\-:]+(\|[\s\-:]+)*\|?$', stripped):
            continue
        cells = [c.strip() for c in stripped.split("|")]
        cells = [c for c in cells if c is not None]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            rows_data.append(cells)

    if not rows_data:
        return

    num_cols = max(len(row) for row in rows_data)
    for row in rows_data:
        while len(row) < num_cols:
            row.append("")

    # Calculate column widths
    avail_w = pdf.w - pdf.l_margin - pdf.r_margin
    col_w = avail_w / num_cols

    for row_idx, row_data in enumerate(rows_data):
        if row_idx == 0:
            # Header row
            pdf.set_fill_color(*COLORS["table_header"])
            pdf.set_text_color(*COLORS["white"])
            pdf.set_font("helvetica", "B", 9)
        else:
            if row_idx % 2 == 0:
                pdf.set_fill_color(*COLORS["table_alt_row"])
            else:
                pdf.set_fill_color(*COLORS["white"])
            pdf.set_text_color(*COLORS["body"])
            pdf.set_font("helvetica", "", 9)

        for col_idx, cell_text in enumerate(row_data):
            pdf.cell(col_w, 7, _sanitize_text(cell_text), border=1, fill=True)

        pdf.ln()

    pdf.ln(4)
    pdf.set_text_color(*COLORS["body"])


def _render_inline_text(pdf: ProfessionalPDF, text: str, base_size: int = 11) -> None:
    """
    Render a line of text with inline bold/italic markdown.
    Handles **bold**, *italic*, and plain text segments.
    """
    pattern = re.compile(
        r'(\*\*\*(.+?)\*\*\*)'    # bold italic
        r'|(\*\*(.+?)\*\*)'       # bold
        r'|(\*(.+?)\*)'           # italic
    )

    last_end = 0
    for match in pattern.finditer(text):
        # Plain text before match
        if match.start() > last_end:
            plain = text[last_end:match.start()]
            pdf.set_font("helvetica", "", base_size)
            pdf.write(6, _sanitize_text(plain))

        if match.group(2):  # ***bold italic***
            pdf.set_font("helvetica", "BI", base_size)
            pdf.write(6, _sanitize_text(match.group(2)))
        elif match.group(4):  # **bold**
            pdf.set_font("helvetica", "B", base_size)
            pdf.write(6, _sanitize_text(match.group(4)))
        elif match.group(6):  # *italic*
            pdf.set_font("helvetica", "I", base_size)
            pdf.write(6, _sanitize_text(match.group(6)))

        last_end = match.end()

    # Remaining text
    if last_end < len(text):
        pdf.set_font("helvetica", "", base_size)
        pdf.write(6, _sanitize_text(text[last_end:]))

    pdf.ln(6)


def _render_section_content(pdf: ProfessionalPDF, content: str) -> None:
    """
    Parse section content with markdown-like syntax and render into the PDF.
    Supports: sub-headings, bold, italic, tables, bullets, numbered lists,
    blockquotes, horizontal rules, and ```chart blocks.
    """
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Check for page space — add new page if near bottom
        if pdf.get_y() > pdf.h - 30:
            pdf.add_page()

        # ```chart block
        if line.startswith("```chart"):
            i += 1
            chart_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                chart_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            chart_text = "\n".join(chart_lines)
            chart_data = _parse_chart_block(chart_text)
            if chart_data:
                _add_chart_to_pdf(pdf, chart_data)
            continue

        # --- Horizontal rule
        if re.match(r'^(---+|___+|\*\*\*+)$', line):
            y = pdf.get_y() + 4
            pdf.set_draw_color(*COLORS["border"])
            pdf.line(pdf.l_margin + 20, y, pdf.w - pdf.r_margin - 20, y)
            pdf.ln(10)
            i += 1
            continue

        # ### Sub-sub-heading (H3)
        if line.startswith("### "):
            heading_text = line[4:].strip()
            pdf.ln(4)
            pdf.set_font("helvetica", "B", 12)
            pdf.set_text_color(*COLORS["heading_3"])
            pdf.cell(0, 8, _sanitize_text(heading_text), ln=1)
            pdf.set_text_color(*COLORS["body"])
            pdf.ln(2)
            i += 1
            continue

        # ## Sub-heading (H2)
        if line.startswith("## "):
            heading_text = line[3:].strip()
            pdf.ln(6)
            pdf.set_font("helvetica", "B", 14)
            pdf.set_text_color(*COLORS["heading_2"])
            pdf.cell(0, 9, _sanitize_text(heading_text), ln=1)
            pdf.set_text_color(*COLORS["body"])
            pdf.ln(3)
            i += 1
            continue

        # > Blockquote
        if line.startswith("> "):
            quote_text = line[2:].strip()
            pdf.ln(2)
            # Draw left border
            x_start = pdf.l_margin + 5
            y_start = pdf.get_y()
            pdf.set_x(pdf.l_margin + 12)
            pdf.set_font("helvetica", "I", 11)
            pdf.set_text_color(*COLORS["blockquote"])
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 15, 6, _sanitize_text(quote_text))
            y_end = pdf.get_y()
            # Left accent bar
            pdf.set_draw_color(*COLORS["accent"])
            pdf.set_line_width(1.5)
            pdf.line(x_start, y_start, x_start, y_end)
            pdf.set_line_width(0.2)
            pdf.set_text_color(*COLORS["body"])
            pdf.ln(4)
            i += 1
            continue

        # | Table row
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            _render_table(pdf, table_lines)
            continue

        # - Bullet point
        if line.startswith("- ") or line.startswith("* "):
            bullet_text = line[2:].strip()
            pdf.set_font("helvetica", "", 11)
            pdf.set_text_color(*COLORS["body"])
            x = pdf.l_margin
            pdf.set_x(x + 5)
            pdf.set_font("helvetica", "", 11)
            pdf.cell(5, 6, chr(149))  # bullet character
            _render_inline_text(pdf, bullet_text)
            i += 1
            continue

        # 1. Numbered list
        if re.match(r'^\d+\.\s+', line):
            num_match = re.match(r'^(\d+)\.\s+', line)
            num = num_match.group(1) if num_match else "1"
            text = re.sub(r'^\d+\.\s+', '', line)
            pdf.set_font("helvetica", "B", 11)
            pdf.set_text_color(*COLORS["accent"])
            pdf.set_x(pdf.l_margin + 5)
            pdf.cell(8, 6, f"{num}.")
            pdf.set_font("helvetica", "", 11)
            pdf.set_text_color(*COLORS["body"])
            _render_inline_text(pdf, text)
            i += 1
            continue

        # Regular paragraph with inline markdown
        pdf.set_text_color(*COLORS["body"])
        _render_inline_text(pdf, line)
        pdf.ln(2)
        i += 1


# ---------------------------------------------------------------------------
# Section divider
# ---------------------------------------------------------------------------


def _add_section_divider(pdf: ProfessionalPDF) -> None:
    """Add a subtle divider between sections."""
    pdf.ln(4)
    y = pdf.get_y()
    center = pdf.w / 2
    pdf.set_draw_color(*COLORS["border"])
    pdf.set_line_width(0.3)
    pdf.line(center - 50, y, center + 50, y)
    pdf.set_line_width(0.2)
    pdf.ln(6)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


async def create_pdf_db(
    input_data: CreatePDFInput,
    sql_db: Session,
    user_id: str,
    conversation_id: str,
) -> str:
    """Generate a professional PDF file from structured section data and persist a DB record."""
    try:
        # 1. Validate user / org
        user = sql_db.query(User).filter(User.id == user_id).first()
        if not user or not user.org_id:
            return "❌ User or Organization not found."

        # 2. Validate section count
        if len(input_data.sections) > MAX_SECTIONS:
            return f"❌ Too many sections ({len(input_data.sections)}). Maximum is {MAX_SECTIONS}."

        # 3. Prepare output path
        sanitized_title = re.sub(r'[^\w\s-]', '', input_data.title).strip().replace(' ', '_')
        if not sanitized_title:
            sanitized_title = "document"

        unique_suffix = str(uuid.uuid4())[:8]
        filename = f"{sanitized_title}_{unique_suffix}.pdf"

        storage_rel = f"storage/pdfs/{filename}"
        os.makedirs("storage/pdfs", exist_ok=True)

        # 4. Build PDF document
        pdf = ProfessionalPDF(doc_title=input_data.title)

        # -- Cover page
        _add_cover_page(pdf, input_data.title)

        # -- Table of Contents (only if 3+ sections)
        if len(input_data.sections) >= 3:
            _add_table_of_contents(pdf, input_data.sections)

        # -- Render each section
        for idx, sec in enumerate(input_data.sections):
            pdf._is_cover_page = False
            pdf.add_page()

            # Section heading
            pdf.set_font("helvetica", "B", 18)
            pdf.set_text_color(*COLORS["heading_1"])
            pdf.cell(0, 12, _sanitize_text(sec.title), ln=1)
            # Underline accent
            y = pdf.get_y()
            pdf.set_draw_color(*COLORS["accent"])
            pdf.set_line_width(0.8)
            pdf.line(pdf.l_margin, y, pdf.l_margin + 60, y)
            pdf.set_line_width(0.2)
            pdf.ln(8)

            # Render content with full markdown support
            pdf.set_text_color(*COLORS["body"])
            pdf.set_font("helvetica", "", 11)
            _render_section_content(pdf, sec.content)

        # 5. Save
        pdf.output(storage_rel)

        # 6. Check file size
        file_size = os.path.getsize(storage_rel)
        if file_size > MAX_FILE_SIZE_BYTES:
            os.remove(storage_rel)
            return (
                f"❌ Generated file is {file_size / (1024 * 1024):.1f} MB which "
                f"exceeds the 10 MB limit. Please reduce the content."
            )

        # 7. Build public URL
        base = input_data.base_url.rstrip("/")
        file_url = f"{base}/{storage_rel}"

        # 8. Persist DB record
        record = GeneratedPDF(
            conversation_id=conversation_id,
            org_id=user.org_id,
            title=input_data.title,
            file_path=storage_rel,
            file_url=file_url,
        )
        sql_db.add(record)
        sql_db.flush()

        for i, sec_in in enumerate(input_data.sections):
            section_record = PDFSection(
                generated_pdf_id=record.id,
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
            f"✅ PDF **'{record.title}'** created successfully!\n"
            f"📄 Sections: {section_names}\n"
            f"📥 **Download link:** {file_url}"
        )

    except Exception as e:
        sql_db.rollback()
        logger.error("Error creating PDF: %s", e, exc_info=True)
        return f"❌ Error creating PDF: {str(e)}"


def get_pdf_link_db(sql_db: Session, conversation_id: str) -> str:
    """Return the download link for the latest PDF in this conversation."""
    try:
        record = (
            sql_db.query(GeneratedPDF)
            .filter(GeneratedPDF.conversation_id == conversation_id)
            .order_by(GeneratedPDF.created_at.desc())
            .first()
        )
        if not record:
            return "📋 No PDF found for this conversation. Use 'create_pdf' to generate one."
        if not record.file_url:
            return "📋 PDF record found but the file has not been generated yet."
        return (
            f"📥 **PDF:** '{record.title}'\n"
            f"**Download:** {record.file_url}"
        )
    except Exception as e:
        return f"❌ Error retrieving PDF link: {str(e)}"


# ---------------------------------------------------------------------------
# StructuredTool integration (LangChain)
# ---------------------------------------------------------------------------


def pdf_generation_tool(
    sql_db: Session,
    user_id: str,
    conversation_id: Optional[str] = None,
    base_url: str = "http://localhost:8000",
) -> List[StructuredTool]:
    """Returns LangChain StructuredTools for PDF generation."""

    if not conversation_id:
        logger.warning("pdf_generation_tool called without conversation_id.")
        return []

    async def create_pdf(title: str, sections: List[Dict[str, Any]]) -> str:
        """
        Generate a professional PDF document from structured section data
        and return a download link.

        The document will include a cover page, table of contents, professional
        formatting, headers and footers with page numbers.

        CONTENT FORMATTING — the 'content' field supports markdown-like syntax:
          - **bold text** → bold
          - *italic text* → italic
          - ## Sub-Heading → Heading Level 2
          - ### Sub-Sub-Heading → Heading Level 3
          - - item → bullet point
          - 1. item → numbered list
          - > text → blockquote with left border
          - --- → horizontal rule / section divider
          - | Col1 | Col2 | → table (use | separators, add |---|---| for header row)
          - ```chart ... ``` → embedded chart (bar, line, or pie)

        CHART FORMAT (inside content):
          ```chart
          type: bar
          title: Revenue by Quarter
          data:
            Q1: 180
            Q2: 210
            Q3: 245
            Q4: 285
          ```
          Supported types: bar, line, pie

        SECTIONS FORMAT — each item must have:
          - "title"   : section heading (str)
          - "content" : section body text (str, supports above markdown formatting)

        LIMITS:
          - Max 50 sections
          - Max 10 MB file size
        """
        parsed_sections = [
            PDFSectionInput(**s) if isinstance(s, dict) else s for s in sections
        ]
        input_obj = CreatePDFInput(
            title=title,
            sections=parsed_sections,
            base_url=base_url,
        )
        return await create_pdf_db(input_obj, sql_db, user_id, conversation_id)

    def get_pdf_link() -> str:
        """Return the download link for the most recently created PDF in this conversation."""
        return get_pdf_link_db(sql_db, conversation_id)

    class _CreateInput(BaseModel):
        title: str = Field(description="Title / label for the PDF document.")
        sections: List[Dict[str, Any]] = Field(
            description=(
                "List of section objects. Each must have 'title' (str) and "
                "'content' (str, supports markdown: **bold**, *italic*, ## headings, "
                "tables, bullets, blockquotes, and ```chart blocks for bar/line/pie charts). "
                "Optional 'section_type' (str). "
                "Max 50 sections. Max 10 MB file size."
            )
        )

    return [
        StructuredTool.from_function(
            coroutine=create_pdf,
            name="create_pdf",
            description=(
                "Generate a professional PDF document with cover page, table of contents, "
                "charts, and rich formatting. Pass 'title' (document name) and 'sections' (list of "
                "{title, content} objects). Content supports markdown: **bold**, *italic*, "
                "## sub-headings, | tables |, - bullets, > blockquotes, --- rules, and "
                "```chart blocks (bar, line, pie). "
                "Returns a download link on success. Limits: max 50 sections, 10 MB."
            ),
            args_schema=_CreateInput,
        ),
        StructuredTool.from_function(
            func=get_pdf_link,
            name="get_pdf_link",
            description=(
                "Retrieve the download link for the most recent PDF "
                "created in this conversation."
            ),
            args_schema=None,
        ),
    ]
