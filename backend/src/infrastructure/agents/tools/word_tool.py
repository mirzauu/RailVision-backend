"""
Word Document Management Tool for storing document sections in the database.

PAGE FORMAT SPECIFICATION (A4 — 8.27" × 11.69")
=================================================
Each section = exactly 1 page.  Content that overflows BREAKS the viewer.

CONTENT BUDGET PER SECTION (PAGE):
  • Plain paragraphs only        → max 450 words  / ~2 800 chars
  • Paragraphs + 1 table (3-col) → max 200 words  (table takes vertical space)
  • Paragraphs + 1 list (8 items)→ max 300 words
  • Paragraphs + table + list    → max 150 words

ELEMENT LIMITS (hard limits enforced below):
  Document title     → 50 chars
  Section title      → 55 chars
  Paragraph          → 65 words / ~400 chars
  ### Sub-heading    → 45 chars
  #### Sub-sub       → 50 chars
  Bullet / Number    → 70 chars per item, max 8 items, 1 level deep
  Table columns      → max 4 cols  (2-col=55, 3-col=35, 4-col=22 chars/cell)
  Table rows         → max 8 rows
  Table header cell  → 20 chars
  Code line          → 80 chars / max 20 lines
"""

import re
import uuid
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

# Import project-specific models
from src.infrastructure.database.models import User, GeneratedWord, WordSection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())

def _truncate(text: str, max_chars: int) -> str:
    """Hard-truncate a string and append ellipsis if trimmed."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"

def _validate_content(content: str) -> str:
    """
    Validate and lightly sanitise section content so it fits on one A4 page.

    Rules applied (in order):
      1. Strip leading/trailing whitespace.
      2. Reject / truncate headings that exceed character limits.
      3. Enforce paragraph word limits (65 words each).
      4. Enforce table constraints (max 4 cols, header 20 chars, cell 35 chars, 8 rows).
      5. Enforce list item limits (max 8 items, 70 chars each, 1 nesting level).
      6. Enforce code-block limits (80 chars/line, 20 lines).
      7. Enforce total content word budget (450 words plain / warn if table or list present).
    """
    lines = content.split("\n")
    output_lines: List[str] = []
    in_code_block = False
    code_lines: List[str] = []
    in_table = False
    table_rows: List[str] = []
    tables_found = 0
    lists_found = 0
    list_item_count = 0
    total_word_count = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Code block ──────────────────────────────────────────────────────
        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lines = [line]
            else:
                in_code_block = False
                # Trim to 20 lines max (excluding fence lines)
                fence_open = code_lines[0]
                inner = code_lines[1:]
                inner = inner[:20]
                # Trim each line to 80 chars
                inner = [l[:80] for l in inner]
                output_lines.append(fence_open)
                output_lines.extend(inner)
                output_lines.append(line)  # closing ```
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line[:80])
            i += 1
            continue

        # ── Table ────────────────────────────────────────────────────────────
        if stripped.startswith("|") and "|" in stripped[1:]:
            if not in_table:
                in_table = True
                table_rows = []
                lists_found_before = lists_found  # snapshot
            table_rows.append(line)
            i += 1
            # Peek ahead — if next line is also a table row keep collecting
            if i < len(lines) and (lines[i].strip().startswith("|")):
                continue
            else:
                # Flush table
                in_table = False
                tables_found += 1
                flushed = _process_table(table_rows)
                output_lines.extend(flushed)
                table_rows = []
            continue

        if in_table:
            table_rows.append(line)
            i += 1
            continue

        # ── ### / #### headings ─────────────────────────────────────────────
        h3 = re.match(r'^(#{1,2})\s+(.+)', stripped)   # # or ## inside content — remap to ###
        h_sub = re.match(r'^(###)\s+(.+)', stripped)
        h_subsub = re.match(r'^(####)\s+(.+)', stripped)

        if h_subsub:
            text = _truncate(h_subsub.group(2).strip(), 50)
            output_lines.append(f"#### {text}")
            i += 1
            continue

        if h_sub:
            text = _truncate(h_sub.group(2).strip(), 45)
            output_lines.append(f"### {text}")
            i += 1
            continue

        # # or ## → downgrade to ### (they are reserved for section titles)
        if h3:
            text = _truncate(h3.group(2).strip(), 45)
            output_lines.append(f"### {text}")
            i += 1
            continue

        # ── Lists ────────────────────────────────────────────────────────────
        bullet = re.match(r'^(\s*)([-*])\s+(.+)', line)
        numbered = re.match(r'^(\s*)(\d+\.)\s+(.+)', line)
        list_match = bullet or numbered

        if list_match:
            indent = list_match.group(1)
            marker = list_match.group(2)
            item_text = list_match.group(3)

            # Only 1 level of nesting allowed — flatten deeper levels
            if len(indent) > 2:
                indent = "  "

            # Count top-level items per list block
            if len(indent) == 0:
                list_item_count += 1
            if list_item_count > 8:
                i += 1
                continue  # drop excess items

            item_text = _truncate(item_text, 70)
            output_lines.append(f"{indent}{marker} {item_text}")
            total_word_count += _word_count(item_text)
            i += 1
            continue

        # Reset list item counter when we exit a list block
        if not list_match and list_item_count > 0:
            list_item_count = 0
            lists_found += 1

        # ── Divider / blockquote (pass-through) ─────────────────────────────
        if stripped == "---" or stripped.startswith(">"):
            output_lines.append(line)
            i += 1
            continue

        # ── Normal paragraph line ────────────────────────────────────────────
        if stripped:
            words = stripped.split()
            # If this paragraph would exceed 65 words, trim it
            if len(words) > 65:
                words = words[:65]
                stripped = " ".join(words) + "…"
            output_lines.append(stripped)
            total_word_count += len(words)
        else:
            output_lines.append("")

        i += 1

    # ── Global word-budget guard ─────────────────────────────────────────────
    has_table = tables_found > 0
    has_list = lists_found > 0

    if has_table and has_list:
        max_words = 150
    elif has_table:
        max_words = 200
    elif has_list:
        max_words = 300
    else:
        max_words = 450

    # If we are already under budget, fine.  Otherwise trim trailing paragraphs.
    if total_word_count > max_words:
        output_lines = _trim_to_budget(output_lines, max_words)

    return "\n".join(output_lines).strip()


def _process_table(rows: List[str]) -> List[str]:
    """Enforce table constraints: max 4 cols, header 20 chars, cell 35 chars, 8 rows."""
    MAX_COLS = 4
    MAX_ROWS = 8  # excluding separator row
    HEADER_MAX = 20
    CELL_MAX = 35

    out = []
    data_rows = 0

    for row in rows:
        # Separator row (|---|---|)
        if re.match(r'^\s*\|[\s\-|]+\|\s*$', row):
            out.append(row)
            continue

        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        # Trim to max 4 cols
        cells = cells[:MAX_COLS]

        # Decide if this is a header row (first non-separator row)
        if data_rows == 0:
            cells = [_truncate(c, HEADER_MAX) for c in cells]
        else:
            if data_rows > MAX_ROWS:
                continue  # drop excess rows
            cells = [_truncate(c, CELL_MAX) for c in cells]

        data_rows += 1
        out.append("| " + " | ".join(cells) + " |")

    return out


def _trim_to_budget(lines: List[str], max_words: int) -> List[str]:
    """Remove trailing lines to keep content within the word budget."""
    running = 0
    result = []
    for line in lines:
        words = len(line.split())
        if running + words > max_words:
            break
        result.append(line)
        running += words
    return result


# --- Tool Input Schemas ---

class CreateWordInput(BaseModel):
    title: str = Field(
        description=(
            "Title of the Word document. "
            "HARD LIMIT: 50 characters maximum. Keep it concise."
        )
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        return _truncate(v.strip(), 50)


class AddWordSectionInput(BaseModel):
    title: str = Field(
        description=(
            "Title of the section (= one A4 page). "
            "HARD LIMIT: 55 characters maximum."
        )
    )
    content: str = Field(
        description=(
            "Markdown content for this section. One section = one A4 page. "
            "HARD PAGE BUDGET — choose one scenario:\n"
            "  • Plain text only         → max 450 words total\n"
            "  • Text + 1 table (3 cols) → max 200 words of text\n"
            "  • Text + 1 list (8 items) → max 300 words of text\n"
            "  • Text + table + list     → max 150 words of text\n"
            "ELEMENT LIMITS:\n"
            "  - Paragraph: max 65 words, 4–5 sentences\n"
            "  - ### Sub-heading: max 45 chars  |  #### Sub-sub: max 50 chars\n"
            "  - Bullet/numbered item: max 70 chars, max 8 items, 1 nesting level\n"
            "  - Table: max 4 cols, max 8 rows, header ≤ 20 chars, cell ≤ 35 chars\n"
            "  - Code block: max 80 chars/line, max 20 lines\n"
            "DO NOT use # or ## (reserved for document/section titles).\n"
            "NEVER pack more than one logical topic on a single page/section."
        )
    )
    section_type: str = Field(
        default="text",
        description="Type of section: text, list, table"
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        return _truncate(v.strip(), 55)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        return _validate_content(v)


class UpdateWordInput(BaseModel):
    title: Optional[str] = Field(
        default=None,
        description="New title for the Word document (max 50 chars)."
    )
    sections: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=(
            "Complete list of sections to replace current ones. "
            "Each dict must have 'title' (max 55 chars), 'content' (see page budget), and 'section_type'. "
            "Each section = 1 A4 page — keep content within budget."
        )
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: Optional[str]) -> Optional[str]:
        if v:
            return _truncate(v.strip(), 50)
        return v

    @field_validator("sections")
    @classmethod
    def validate_sections(cls, v: Optional[List[Dict[str, str]]]) -> Optional[List[Dict[str, str]]]:
        if not v:
            return v
        validated = []
        for s in v:
            s = dict(s)
            if "title" in s:
                s["title"] = _truncate(s["title"].strip(), 55)
            if "content" in s:
                s["content"] = _validate_content(s["content"])
            validated.append(s)
        return validated


# --- Core Logic Functions (Database interaction) ---

async def create_word_db(input_data: CreateWordInput, sql_db: Session, user_id: str, conversation_id: str) -> str:
    """Initialize a new generated Word document in the database"""
    try:
        user = sql_db.query(User).filter(User.id == user_id).first()
        if not user or not user.org_id:
            return "❌ User or Organization not found."

        word_doc = GeneratedWord(
            conversation_id=conversation_id,
            org_id=user.org_id,
            title=input_data.title
        )
        sql_db.add(word_doc)
        sql_db.commit()
        sql_db.refresh(word_doc)
        
        return (
            f"✅ Created new Word document: '{word_doc.title}'.\n"
            f"REMINDER — each section = 1 A4 page. Keep content within the page budget:\n"
            f"  • Plain text only: max 450 words\n"
            f"  • Text + table: max 200 words\n"
            f"  • Text + list: max 300 words\n"
            f"  • Text + table + list: max 150 words\n"
            f"Add sections now using 'add_word_section'."
        )
    except Exception as e:
        sql_db.rollback()
        return f"❌ Error creating Word document in DB: {str(e)}"

async def add_word_section_db(input_data: AddWordSectionInput, sql_db: Session, conversation_id: str) -> str:
    """Add a section to the current Word document in the database"""
    try:
        # Find the latest Word document for this conversation
        word_doc = sql_db.query(GeneratedWord).filter(GeneratedWord.conversation_id == conversation_id).order_by(GeneratedWord.created_at.desc()).first()
        if not word_doc:
            return "❌ No Word document found for this conversation. Please create one first using 'create_word_doc'."

        # Get current section count for ordering
        section_count = sql_db.query(WordSection).filter(WordSection.generated_word_id == word_doc.id).count()
        
        section = WordSection(
            generated_word_id=word_doc.id,
            title=input_data.title,
            content=input_data.content,
            section_type=input_data.section_type,
            order=section_count + 1
        )
        sql_db.add(section)
        sql_db.commit()
        
        word_count = _word_count(input_data.content)
        return (
            f"✅ Added section '{input_data.title}' to '{word_doc.title}'. "
            f"Total sections: {section_count + 1}. "
            f"Section word count: {word_count}."
        )
    except Exception as e:
        sql_db.rollback()
        return f"❌ Error adding section to DB: {str(e)}"

async def update_word_db(input_data: UpdateWordInput, sql_db: Session, conversation_id: str) -> str:
    """Update the current Word document in the database"""
    try:
        word_doc = sql_db.query(GeneratedWord).filter(GeneratedWord.conversation_id == conversation_id).order_by(GeneratedWord.created_at.desc()).first()
        if not word_doc:
            return "❌ No Word document found to update."

        if input_data.title:
            word_doc.title = input_data.title
        
        if input_data.sections:
            # Delete existing sections and replace
            sql_db.query(WordSection).filter(WordSection.generated_word_id == word_doc.id).delete()
            for i, s_data in enumerate(input_data.sections):
                section = WordSection(
                    generated_word_id=word_doc.id,
                    title=s_data.get('title', ''),
                    content=s_data.get('content', ''),
                    section_type=s_data.get('section_type', 'text'),
                    order=i + 1
                )
                sql_db.add(section)
        
        sql_db.commit()
        return f"✅ Updated Word document '{word_doc.title}' successfully."
    except Exception as e:
        sql_db.rollback()
        return f"❌ Error updating Word document: {str(e)}"

def list_word_sections_db(sql_db: Session, conversation_id: str) -> str:
    """List sections of the current Word document in the database"""
    try:
        word_doc = sql_db.query(GeneratedWord).filter(GeneratedWord.conversation_id == conversation_id).order_by(GeneratedWord.created_at.desc()).first()
        if not word_doc:
            return "📋 No Word document found for this conversation."
        
        sections = sql_db.query(WordSection).filter(WordSection.generated_word_id == word_doc.id).order_by(WordSection.order.asc()).all()
        if not sections:
            return f"📋 Word document '{word_doc.title}' has no sections yet."
        
        result = f"📋 **Word Document: {word_doc.title}** ({len(sections)} sections)\n\n"
        for section in sections:
            wc = _word_count(section.content)
            result += f"{section.order}. **{section.title}** ({section.section_type}) — {wc} words\n"
        return result
    except Exception as e:
        return f"❌ Error retrieving sections: {str(e)}"

# --- Integration for Project (StructuredTool) ---

def word_generation_tool(sql_db: Session, user_id: str, conversation_id: Optional[str] = None) -> List[StructuredTool]:
    """Returns tools in StructuredTool format for project integration"""
    
    if not conversation_id:
        import logging
        logging.warning("word_generation_tool called without conversation_id. Word tools will not be available.")
        return []

    async def create_word_doc(title: str) -> str:
        """Initialize a new Word document in the database. Title max 50 chars."""
        return await create_word_db(CreateWordInput(title=title), sql_db, user_id, conversation_id)

    async def add_word_section(title: str, content: str, section_type: str = "text") -> str:
        """
        Add ONE section (= ONE A4 page) to the Word document.

        PAGE BUDGET — pick ONE scenario and stay within it:
          • Plain text only           → max 450 words
          • Text + 1 table (≤3 cols) → max 200 words of surrounding text
          • Text + 1 list (≤8 items) → max 300 words of surrounding text
          • Text + table + list       → max 150 words of surrounding text

        ELEMENT HARD LIMITS:
          title          : max 55 chars
          paragraph      : max 65 words / 4–5 sentences
          ### heading    : max 45 chars
          #### heading   : max 50 chars
          bullet/number  : max 70 chars per item, max 8 items, 1 nesting level
          table cols     : max 4  |  table rows: max 8
          table header   : max 20 chars per cell
          table cell     : 2-col=55, 3-col=35, 4-col=22 chars
          code line      : max 80 chars / max 20 lines total

        Split content across MULTIPLE sections if it is long.
        NEVER put more than one logical topic on a single section/page.
        """
        return await add_word_section_db(
            AddWordSectionInput(title=title, content=content, section_type=section_type),
            sql_db,
            conversation_id
        )

    def list_word_sections() -> str:
        """List all sections currently in the database for the current Word document draft."""
        return list_word_sections_db(sql_db, conversation_id)

    async def update_word_doc(title: Optional[str] = None, sections: Optional[List[Dict[str, str]]] = None) -> str:
        """Update the existing Word document in the database. Each section must fit in one A4 page."""
        return await update_word_db(UpdateWordInput(title=title, sections=sections), sql_db, conversation_id)
    
    return [
        StructuredTool.from_function(
            coroutine=create_word_doc,
            name="create_word_doc",
            description=(
                "Initialize a new Word document record in the database. "
                "Use this FIRST. Title must be max 50 characters."
            ),
            args_schema=CreateWordInput
        ),
        StructuredTool.from_function(
            coroutine=add_word_section,
            name="add_word_section",
            description=(
                "Add ONE section (= ONE A4 page) to the current Word document. "
                "STRICT PAGE BUDGET per section: "
                "plain text → max 450 words; "
                "text + table → max 200 words; "
                "text + list → max 300 words; "
                "text + table + list → max 150 words. "
                "Split long content across MULTIPLE sections. "
                "Element limits: paragraph ≤65 words, ### heading ≤45 chars, "
                "bullet items ≤70 chars (max 8), table ≤4 cols / ≤8 rows / header ≤20 chars / cell ≤35 chars, "
                "code ≤80 chars/line (max 20 lines). "
                "DO NOT use # or ## headings — only ### and #### inside section content."
            ),
            args_schema=AddWordSectionInput
        ),
        StructuredTool.from_function(
            func=list_word_sections,
            name="list_word_sections",
            description="List all sections stored in the database for the current Word document.",
            args_schema=None
        ),
        StructuredTool.from_function(
            coroutine=update_word_doc,
            name="update_word_doc",
            description=(
                "Update the current Word document in the database (title or sections). "
                "Each section must still respect the A4 page budget."
            ),
            args_schema=UpdateWordInput
        )
    ]
