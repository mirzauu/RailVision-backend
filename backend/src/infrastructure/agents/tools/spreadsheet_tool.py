"""
Spreadsheet Tool — generates Excel (.xlsx) files from structured sheet data.

Architecture:
  User → Agent → create_spreadsheet tool
                    ↓
                SheetData (name + rows list[dict])
                    ↓
                pandas DataFrame
                    ↓
                pd.ExcelWriter  (openpyxl engine)
                    ↓
                storage/spreadsheets/<uuid>.xlsx
                    ↓
                Download URL

Resource limits (enforced here, not by the LLM):
  - Max sheets      : 10
  - Max rows/sheet  : 50 000
  - Max columns     : 100
  - Max file size   : 10 MB
"""

import os
import uuid
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from src.infrastructure.database.models import User, GeneratedSpreadsheet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------

MAX_SHEETS = 10
MAX_ROWS_PER_SHEET = 50_000
MAX_COLUMNS = 100
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# Input Schemas
# ---------------------------------------------------------------------------


class SheetData(BaseModel):
    """Represents one worksheet in the spreadsheet."""

    name: str = Field(
        description=(
            "Name for this worksheet tab. "
            "Must be ≤ 31 characters (Excel limit). "
            "Avoid special characters: \\ / ? * [ ]"
        )
    )
    rows: List[Dict[str, Any]] = Field(
        description=(
            "List of row objects. Each dict key becomes a column header; "
            "each dict value becomes the cell value. "
            "All rows in one sheet should share the same keys. "
            "Max 50 000 rows per sheet."
        )
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Sheet name cannot be empty.")
        # Excel sheet name limit is 31 characters
        if len(v) > 31:
            v = v[:31]
        # Strip characters disallowed by Excel
        for ch in r"\/?*[]":
            v = v.replace(ch, "_")
        return v

    @field_validator("rows")
    @classmethod
    def validate_rows(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(v) > MAX_ROWS_PER_SHEET:
            raise ValueError(
                f"Sheet has {len(v)} rows but the maximum is {MAX_ROWS_PER_SHEET}."
            )
        if v:
            ncols = len(v[0])
            if ncols > MAX_COLUMNS:
                raise ValueError(
                    f"Sheet has {ncols} columns but the maximum is {MAX_COLUMNS}."
                )
        return v


class CreateSpreadsheetInput(BaseModel):
    """Input schema for the create_spreadsheet tool."""

    title: str = Field(
        description="Title / filename label for the spreadsheet (stored in the database)."
    )
    sheets: List[SheetData] = Field(
        description=(
            "List of sheets to include. Each sheet has a name and a list of row dicts. "
            "Maximum 10 sheets. "
            "Example: [{\"name\": \"Summary\", \"rows\": [{\"Region\": \"North\", \"Revenue\": 120000}]}]"
        )
    )
    base_url: str = Field(
        description=(
            "Base URL of the backend server (e.g. http://localhost:8000). "
            "The tool appends the file path to form the download link. "
            "This value is automatically injected by the agent — do not ask the user for it."
        )
    )

    @field_validator("sheets")
    @classmethod
    def validate_sheets(cls, v: List[SheetData]) -> List[SheetData]:
        if not v:
            raise ValueError("At least one sheet is required.")
        if len(v) > MAX_SHEETS:
            raise ValueError(
                f"{len(v)} sheets provided but the maximum is {MAX_SHEETS}."
            )
        # Enforce unique sheet names
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate sheet names are not allowed.")
        return v


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _auto_fit_columns(worksheet) -> None:  # type: ignore[type-arg]
    """Best-effort column auto-width for an openpyxl worksheet."""
    from openpyxl.utils import get_column_letter

    for col in worksheet.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                if cell.value:
                    max_len = max(max_len, len(str(cell.value)))
            except Exception:
                pass
        adjusted = min(max_len + 4, 60)  # cap at 60 chars wide
        worksheet.column_dimensions[col_letter].width = adjusted


async def create_spreadsheet_db(
    input_data: CreateSpreadsheetInput,
    sql_db: Session,
    user_id: str,
    conversation_id: str,
) -> str:
    """Generate an .xlsx file from structured sheet data and persist a DB record."""
    try:
        # 1. Validate user / org
        user = sql_db.query(User).filter(User.id == user_id).first()
        if not user or not user.org_id:
            return "❌ User or Organization not found."

        # 2. Prepare output directory and path
        # Sanitize title for filename
        import re
        sanitized_title = re.sub(r'[^\w\s-]', '', input_data.title).strip().replace(' ', '_')
        if not sanitized_title:
            sanitized_title = "spreadsheet"
        
        # Add a short unique suffix to avoid collisions while keeping the name recognizable
        unique_suffix = str(uuid.uuid4())[:8]
        filename = f"{sanitized_title}_{unique_suffix}.xlsx"
        
        storage_rel = f"storage/spreadsheets/{filename}"
        os.makedirs("storage/spreadsheets", exist_ok=True)

        # 3. Build DataFrames and write Excel file
        with pd.ExcelWriter(storage_rel, engine="openpyxl") as writer:
            for sheet in input_data.sheets:
                df = pd.DataFrame(sheet.rows)
                df.to_excel(writer, sheet_name=sheet.name, index=False)

            # Auto-fit column widths for all sheets
            for sheet in input_data.sheets:
                ws = writer.sheets[sheet.name]
                _auto_fit_columns(ws)

        # 4. Check file size
        file_size = os.path.getsize(storage_rel)
        if file_size > MAX_FILE_SIZE_BYTES:
            os.remove(storage_rel)
            return (
                f"❌ Generated file is {file_size / (1024 * 1024):.1f} MB which "
                f"exceeds the 10 MB limit. Please reduce the number of rows or sheets."
            )

        # 5. Build public URL
        base = input_data.base_url.rstrip("/")
        file_url = f"{base}/storage/spreadsheets/{filename}"

        # 6. Persist DB record
        record = GeneratedSpreadsheet(
            conversation_id=conversation_id,
            org_id=user.org_id,
            title=input_data.title,
            file_path=storage_rel,
            file_url=file_url,
            sheet_count=len(input_data.sheets),
        )
        sql_db.add(record)
        sql_db.commit()
        sql_db.refresh(record)

        sheet_names = ", ".join(f"'{s.name}'" for s in input_data.sheets)
        total_rows = sum(len(s.rows) for s in input_data.sheets)

        return (
            f"✅ Spreadsheet **'{input_data.title}'** created successfully!\n"
            f"📄 Sheets: {sheet_names}\n"
            f"📊 Total rows: {total_rows:,}\n"
            f"📥 **Download link:** {file_url}"
        )

    except ValueError as ve:
        return f"❌ Validation error: {str(ve)}"
    except Exception as e:
        sql_db.rollback()
        logger.error("Error creating spreadsheet: %s", e, exc_info=True)
        return f"❌ Error creating spreadsheet: {str(e)}"


def get_spreadsheet_link_db(sql_db: Session, conversation_id: str) -> str:
    """Return the download link for the latest spreadsheet in this conversation."""
    try:
        record = (
            sql_db.query(GeneratedSpreadsheet)
            .filter(GeneratedSpreadsheet.conversation_id == conversation_id)
            .order_by(GeneratedSpreadsheet.created_at.desc())
            .first()
        )
        if not record:
            return "📋 No spreadsheet found for this conversation. Use 'create_spreadsheet' to generate one."
        if not record.file_url:
            return "📋 Spreadsheet record found but the file has not been generated yet."
        return (
            f"📥 **Spreadsheet:** '{record.title}'\n"
            f"**Download:** {record.file_url}\n"
            f"({record.sheet_count} sheet(s))"
        )
    except Exception as e:
        return f"❌ Error retrieving spreadsheet link: {str(e)}"


# ---------------------------------------------------------------------------
# StructuredTool integration (LangChain / PydanticAI)
# ---------------------------------------------------------------------------


def spreadsheet_generation_tool(
    sql_db: Session,
    user_id: str,
    conversation_id: Optional[str] = None,
    base_url: str = "http://localhost:8000",
) -> List[StructuredTool]:
    """
    Returns LangChain StructuredTools for spreadsheet generation.

    Parameters
    ----------
    sql_db          : Active SQLAlchemy session.
    user_id         : ID of the authenticated user.
    conversation_id : ID of the current conversation.
    base_url        : Public base URL of the backend (used to build the download link).
    """
    if not conversation_id:
        logger.warning(
            "spreadsheet_generation_tool called without conversation_id. "
            "Spreadsheet tools will not be available."
        )
        return []

    # ------------------------------------------------------------------
    # create_spreadsheet
    # ------------------------------------------------------------------

    async def create_spreadsheet(
        title: str,
        sheets: List[Dict[str, Any]],
    ) -> str:
        """
        Generate an Excel (.xlsx) spreadsheet from structured sheet data and
        return a download link.

        SHEETS FORMAT — each item must have:
          - "name" : worksheet tab name (str, ≤31 chars)
          - "rows" : list of row dicts where keys = column headers

        EXAMPLE:
          sheets = [
            {
              "name": "Revenue",
              "rows": [
                {"Quarter": "Q1", "Region": "North", "Revenue": 340000},
                {"Quarter": "Q1", "Region": "South", "Revenue": 210000}
              ]
            },
            {
              "name": "Headcount",
              "rows": [
                {"Department": "Engineering", "Count": 42},
                {"Department": "Sales",       "Count": 18}
              ]
            }
          ]

        LIMITS:
          - Max 10 sheets
          - Max 50 000 rows per sheet
          - Max 100 columns per sheet
          - Max 10 MB file size
        """
        # Re-validate via Pydantic before passing to DB function
        parsed_sheets = [SheetData(**s) if isinstance(s, dict) else s for s in sheets]
        input_obj = CreateSpreadsheetInput(
            title=title,
            sheets=parsed_sheets,
            base_url=base_url,
        )
        return await create_spreadsheet_db(input_obj, sql_db, user_id, conversation_id)

    # ------------------------------------------------------------------
    # get_spreadsheet_link
    # ------------------------------------------------------------------

    def get_spreadsheet_link() -> str:
        """
        Return the download link for the most recently created spreadsheet
        in this conversation.
        """
        return get_spreadsheet_link_db(sql_db, conversation_id)

    # ------------------------------------------------------------------
    # Register as StructuredTools
    # ------------------------------------------------------------------

    class _CreateInput(BaseModel):
        title: str = Field(description="Title / label for the spreadsheet.")
        sheets: List[Dict[str, Any]] = Field(
            description=(
                "List of sheet objects. Each must have 'name' (str) and "
                "'rows' (list of dicts where keys are column headers). "
                "Max 10 sheets, 50 000 rows/sheet, 100 columns/sheet."
            )
        )

    return [
        StructuredTool.from_function(
            coroutine=create_spreadsheet,
            name="create_spreadsheet",
            description=(
                "Generate an Excel (.xlsx) spreadsheet with one or more sheets. "
                "Pass 'title' (document name) and 'sheets' (list of {name, rows} objects). "
                "Each row is a dict whose keys are column headers. "
                "Returns a download link on success. "
                "Limits: max 10 sheets, 50 000 rows/sheet, 100 columns, 10 MB."
            ),
            args_schema=_CreateInput,
        ),
        StructuredTool.from_function(
            func=get_spreadsheet_link,
            name="get_spreadsheet_link",
            description=(
                "Retrieve the download link for the most recent spreadsheet "
                "created in this conversation."
            ),
            args_schema=None,
        ),
    ]
