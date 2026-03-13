"""
PDF Document Management Tool
"""

import os
import uuid
import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool
from fpdf import FPDF

from src.infrastructure.database.models import User, GeneratedPDF, PDFSection

logger = logging.getLogger(__name__)

# Limits checking
MAX_SECTIONS = 50
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

class PDFSectionInput(BaseModel):
    title: str = Field(description="Title of the section")
    content: str = Field(description="Text content of the section.")
    section_type: str = Field(default="text", description="Type of slide: text, list")

class CreatePDFInput(BaseModel):
    title: str = Field(description="Title of the PDF report")
    sections: List[PDFSectionInput] = Field(description="List of sections for the report")
    base_url: str = Field(description="Base URL of the backend server")

async def create_pdf_db(input_data: CreatePDFInput, sql_db: Session, user_id: str, conversation_id: str) -> str:
    try:
        user = sql_db.query(User).filter(User.id == user_id).first()
        if not user or not user.org_id:
            return "❌ User or Organization not found."

        import re
        sanitized_title = re.sub(r'[^\w\s-]', '', input_data.title).strip().replace(' ', '_')
        if not sanitized_title:
            sanitized_title = "document"
            
        unique_suffix = str(uuid.uuid4())[:8]
        filename = f"{sanitized_title}_{unique_suffix}.pdf"
        
        storage_rel = f"storage/pdfs/{filename}"
        os.makedirs("storage/pdfs", exist_ok=True)
        
        def _sanitize_text(text: str) -> str:
            if not text:
                return ""
            replacements = {
                '“': '"', '”': '"', '‘': "'", '’': "'",
                '—': '-', '–': '-', '…': '...'
            }
            for k, v in replacements.items():
                text = text.replace(k, v)
            return text.encode('latin-1', 'replace').decode('latin-1')

        class PDF(FPDF):
            def header(self):
                self.set_font("helvetica", "B", 15)
                self.cell(0, 10, _sanitize_text(input_data.title), border=False, align="C")
                self.ln(20)
                
            def footer(self):
                self.set_y(-15)
                self.set_font("helvetica", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=11)
        
        for i, section in enumerate(input_data.sections):
            pdf.set_font("helvetica", "B", 14)
            pdf.cell(0, 10, txt=_sanitize_text(section.title), ln=1)
            pdf.set_font("helvetica", size=11)
            pdf.multi_cell(0, 6, txt=_sanitize_text(section.content))
            pdf.ln(5)

        pdf.output(storage_rel)
        
        file_size = os.path.getsize(storage_rel)
        if file_size > MAX_FILE_SIZE_BYTES:
            os.remove(storage_rel)
            return "❌ Generated file is too large. Please reduce the content."

        base = input_data.base_url.rstrip("/")
        file_url = f"{base}/{storage_rel}"
        
        record = GeneratedPDF(
            conversation_id=conversation_id,
            org_id=user.org_id,
            title=input_data.title,
            file_path=storage_rel,
            file_url=file_url
        )
        sql_db.add(record)
        sql_db.flush()
        
        for i, sec_in in enumerate(input_data.sections):
            section_record = PDFSection(
                generated_pdf_id=record.id,
                title=sec_in.title,
                content=sec_in.content,
                section_type=sec_in.section_type,
                order=i+1
            )
            sql_db.add(section_record)
            
        sql_db.commit()
        sql_db.refresh(record)
        
        return (
            f"✅ PDF **'{record.title}'** created successfully!\n"
            f"📥 **Download link:** {file_url}"
        )
    except Exception as e:
        sql_db.rollback()
        logger.error("Error creating PDF: %s", e, exc_info=True)
        return f"❌ Error creating PDF: {str(e)}"

def get_pdf_link_db(sql_db: Session, conversation_id: str) -> str:
    try:
        record = sql_db.query(GeneratedPDF).filter(GeneratedPDF.conversation_id == conversation_id).order_by(GeneratedPDF.created_at.desc()).first()
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

def pdf_generation_tool(
    sql_db: Session, 
    user_id: str, 
    conversation_id: Optional[str] = None, 
    base_url: str = "http://localhost:8000"
) -> List[StructuredTool]:
    """Returns LangChain StructuredTools for PDF generation."""
    
    if not conversation_id:
        logger.warning("pdf_generation_tool called without conversation_id.")
        return []
    
    async def create_pdf(title: str, sections: List[Dict[str, Any]]) -> str:
        """
        Generate a PDF report from structured section data and return a download link.
        """
        parsed_sections = [PDFSectionInput(**s) if isinstance(s, dict) else s for s in sections]
        return await create_pdf_db(CreatePDFInput(title=title, sections=parsed_sections, base_url=base_url), sql_db, user_id, conversation_id)
        
    def get_pdf_link() -> str:
        """Retrieve the download link for the most recent PDF created in this conversation."""
        return get_pdf_link_db(sql_db, conversation_id)
        
    class _CreateInput(BaseModel):
        title: str = Field(description="Title / label for the PDF report.")
        sections: List[Dict[str, Any]] = Field(description="List of section objects. Each must have 'title' (str) and 'content' (str). Optional 'section_type' (str).")

    return [
        StructuredTool.from_function(
            coroutine=create_pdf,
            name="create_pdf",
            description="Generate a PDF report with multiple sections. Pass 'title' (document name) and 'sections' (list of {title, content} objects). Returns a download link on success.",
            args_schema=_CreateInput,
        ),
        StructuredTool.from_function(
            func=get_pdf_link,
            name="get_pdf_link",
            description="Retrieve the download link for the most recent PDF created in this conversation.",
            args_schema=None,
        ),
    ]
