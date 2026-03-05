"""
PDF Management Tool for storing document sections in the database.
"""

import uuid
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

# Import project-specific models
from src.infrastructure.database.models import User, GeneratedPDF, PDFSection

# --- Tool Input Schemas ---

class CreatePDFInput(BaseModel):
    title: str = Field(description="Title of the PDF document")

class AddPDFSectionInput(BaseModel):
    title: str = Field(description="Title of the section")
    content: str = Field(description="Content of the section (Markdown supported)")
    section_type: str = Field(default="text", description="Type of section: text, list, table")

class UpdatePDFInput(BaseModel):
    title: Optional[str] = Field(default=None, description="New title for the PDF")
    sections: Optional[List[Dict[str, str]]] = Field(default=None, description="Complete list of sections to replace current ones. Each dict should have 'title', 'content', and 'section_type'.")

# --- Core Logic Functions (Database interaction) ---

async def create_pdf_db(input_data: CreatePDFInput, sql_db: Session, user_id: str, conversation_id: str) -> str:
    """Initialize a new generated PDF in the database"""
    try:
        user = sql_db.query(User).filter(User.id == user_id).first()
        if not user or not user.org_id:
            return "❌ User or Organization not found."

        pdf = GeneratedPDF(
            conversation_id=conversation_id,
            org_id=user.org_id,
            title=input_data.title
        )
        sql_db.add(pdf)
        sql_db.commit()
        sql_db.refresh(pdf)
        
        return f"✅ Created new PDF: '{pdf.title}' for this conversation. You can now add sections."
    except Exception as e:
        sql_db.rollback()
        return f"❌ Error creating PDF in DB: {str(e)}"

async def add_pdf_section_db(input_data: AddPDFSectionInput, sql_db: Session, conversation_id: str) -> str:
    """Add a section to the current PDF in the database"""
    try:
        # Find the latest PDF for this conversation
        pdf = sql_db.query(GeneratedPDF).filter(GeneratedPDF.conversation_id == conversation_id).order_by(GeneratedPDF.created_at.desc()).first()
        if not pdf:
            return "❌ No PDF found for this conversation. Please create one first using 'create_pdf'."

        # Get current section count for ordering
        section_count = sql_db.query(PDFSection).filter(PDFSection.generated_pdf_id == pdf.id).count()
        
        section = PDFSection(
            generated_pdf_id=pdf.id,
            title=input_data.title,
            content=input_data.content,
            section_type=input_data.section_type,
            order=section_count + 1
        )
        sql_db.add(section)
        sql_db.commit()
        
        return f"✅ Added section '{input_data.title}' to PDF '{pdf.title}'. Total sections: {section_count + 1}"
    except Exception as e:
        sql_db.rollback()
        return f"❌ Error adding section to DB: {str(e)}"

async def update_pdf_db(input_data: UpdatePDFInput, sql_db: Session, conversation_id: str) -> str:
    """Update the current PDF in the database"""
    try:
        pdf = sql_db.query(GeneratedPDF).filter(GeneratedPDF.conversation_id == conversation_id).order_by(GeneratedPDF.created_at.desc()).first()
        if not pdf:
            return "❌ No PDF found to update."

        if input_data.title:
            pdf.title = input_data.title
        
        if input_data.sections:
            # Delete existing sections and replace
            sql_db.query(PDFSection).filter(PDFSection.generated_pdf_id == pdf.id).delete()
            for i, s_data in enumerate(input_data.sections):
                section = PDFSection(
                    generated_pdf_id=pdf.id,
                    title=s_data.get('title', ''),
                    content=s_data.get('content', ''),
                    section_type=s_data.get('section_type', 'text'),
                    order=i + 1
                )
                sql_db.add(section)
        
        sql_db.commit()
        return f"✅ Updated PDF '{pdf.title}' successfully."
    except Exception as e:
        sql_db.rollback()
        return f"❌ Error updating PDF: {str(e)}"

def list_pdf_sections_db(sql_db: Session, conversation_id: str) -> str:
    """List sections of the current PDF in the database"""
    try:
        pdf = sql_db.query(GeneratedPDF).filter(GeneratedPDF.conversation_id == conversation_id).order_by(GeneratedPDF.created_at.desc()).first()
        if not pdf:
            return "📋 No PDF found for this conversation."
        
        sections = sql_db.query(PDFSection).filter(PDFSection.generated_pdf_id == pdf.id).order_by(PDFSection.order.asc()).all()
        if not sections:
            return f"📋 PDF '{pdf.title}' has no sections yet."
        
        result = f"📋 **PDF: {pdf.title}** ({len(sections)} sections)\n\n"
        for section in sections:
            result += f"{section.order}. **{section.title}** ({section.section_type})\n"
        return result
    except Exception as e:
        return f"❌ Error retrieving sections: {str(e)}"

# --- Integration for Project (StructuredTool) ---

def pdf_generation_tool(sql_db: Session, user_id: str, conversation_id: Optional[str] = None) -> List[StructuredTool]:
    """Returns tools in StructuredTool format for project integration"""
    
    if not conversation_id:
        import logging
        logging.warning("pdf_generation_tool called without conversation_id. PDF tools will not be available.")
        return []

    async def create_pdf(title: str) -> str:
        """Initialize a new PDF document in the database."""
        return await create_pdf_db(CreatePDFInput(title=title), sql_db, user_id, conversation_id)

    async def add_pdf_section(title: str, content: str, section_type: str = "text") -> str:
        """Add a section to the current PDF document in the database."""
        return await add_pdf_section_db(AddPDFSectionInput(title=title, content=content, section_type=section_type), sql_db, conversation_id)

    def list_pdf_sections() -> str:
        """List all sections currently in the database for the current PDF draft."""
        return list_pdf_sections_db(sql_db, conversation_id)

    async def update_pdf(title: Optional[str] = None, sections: Optional[List[Dict[str, str]]] = None) -> str:
        """Update the existing PDF in the database."""
        return await update_pdf_db(UpdatePDFInput(title=title, sections=sections), sql_db, conversation_id)
    
    return [
        StructuredTool.from_function(
            coroutine=create_pdf,
            name="create_pdf",
            description="Initialize a new PDF document record in the database. Use this first.",
            args_schema=CreatePDFInput
        ),
        StructuredTool.from_function(
            coroutine=add_pdf_section,
            name="add_pdf_section",
            description="Add a section (text, list, or table) to the current database-stored PDF document.",
            args_schema=AddPDFSectionInput
        ),
        StructuredTool.from_function(
            func=list_pdf_sections,
            name="list_pdf_sections",
            description="List all sections stored in the database for the current PDF.",
            args_schema=None
        ),
        StructuredTool.from_function(
            coroutine=update_pdf,
            name="update_pdf",
            description="Update the current PDF document in the database (title or sections).",
            args_schema=UpdatePDFInput
        )
    ]
