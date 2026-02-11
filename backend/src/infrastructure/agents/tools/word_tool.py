"""
Word Document Management Tool for storing document sections in the database.
"""

import uuid
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

# Import project-specific models
from src.infrastructure.database.models import User, GeneratedWord, WordSection

# --- Tool Input Schemas ---

class CreateWordInput(BaseModel):
    title: str = Field(description="Title of the Word document")

class AddWordSectionInput(BaseModel):
    title: str = Field(description="Title of the section")
    content: str = Field(description="Content of the section (Markdown supported)")
    section_type: str = Field(default="text", description="Type of section: text, list, table")

class UpdateWordInput(BaseModel):
    title: Optional[str] = Field(default=None, description="New title for the Word document")
    sections: Optional[List[Dict[str, str]]] = Field(default=None, description="Complete list of sections to replace current ones. Each dict should have 'title', 'content', and 'section_type'.")

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
        
        return f"✅ Created new Word document: '{word_doc.title}' (ID: {word_doc.id}) for this conversation. You can now add sections."
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
        
        return f"✅ Added section '{input_data.title}' to Word document '{word_doc.title}'. Total sections: {section_count + 1}"
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
        
        result = f"📋 **Word Document: {word_doc.title}** (ID: {word_doc.id}, {len(sections)} sections)\n\n"
        for section in sections:
            result += f"{section.order}. **{section.title}** ({section.section_type})\n"
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
        """Initialize a new Word document in the database."""
        return await create_word_db(CreateWordInput(title=title), sql_db, user_id, conversation_id)

    async def add_word_section(title: str, content: str, section_type: str = "text") -> str:
        """Add a section to the current Word document in the database."""
        return await add_word_section_db(AddWordSectionInput(title=title, content=content, section_type=section_type), sql_db, conversation_id)

    def list_word_sections() -> str:
        """List all sections currently in the database for the current Word document draft."""
        return list_word_sections_db(sql_db, conversation_id)

    async def update_word_doc(title: Optional[str] = None, sections: Optional[List[Dict[str, str]]] = None) -> str:
        """Update the existing Word document in the database."""
        return await update_word_db(UpdateWordInput(title=title, sections=sections), sql_db, conversation_id)
    
    return [
        StructuredTool.from_function(
            coroutine=create_word_doc,
            name="create_word_doc",
            description="Initialize a new Word document record in the database. Use this first.",
            args_schema=CreateWordInput
        ),
        StructuredTool.from_function(
            coroutine=add_word_section,
            name="add_word_section",
            description="Add a section (text, list, or table) to the current database-stored Word document.",
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
            description="Update the current Word document in the database (title or sections).",
            args_schema=UpdateWordInput
        )
    ]
