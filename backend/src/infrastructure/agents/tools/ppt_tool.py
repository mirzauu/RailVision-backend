"""
PowerPoint Management Tool for storing slides in the database.
"""

import uuid
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

# Import project-specific models
from src.infrastructure.database.models import User, Presentation, PresentationSlide

# --- Tool Input Schemas ---

class CreatePPTInput(BaseModel):
    title: str = Field(description="Title of the PowerPoint presentation")

class AddSlideInput(BaseModel):
    title: str = Field(description="Title of the slide")
    content: str = Field(description="Content of the slide. Use newlines for multiple bullet points.")
    slide_type: str = Field(default="bullet", description="Type of slide: bullet, text")

class UpdatePPTInput(BaseModel):
    title: Optional[str] = Field(default=None, description="New title for the PPT")
    slides: Optional[List[Dict[str, str]]] = Field(default=None, description="Complete list of slides to replace current ones. Each dict should have 'title', 'content', and 'slide_type'.")

# --- Core Logic Functions (Database interaction) ---

async def create_ppt_db(input_data: CreatePPTInput, sql_db: Session, user_id: str, conversation_id: str) -> str:
    """Initialize a new PowerPoint presentation in the database"""
    try:
        user = sql_db.query(User).filter(User.id == user_id).first()
        if not user or not user.org_id:
            return "❌ User or Organization not found."

        # Check if conversation already has a PPT, if so, we might want to update or create new
        # For now, let's just create a new one as requested by the flow
        ppt = Presentation(
            conversation_id=conversation_id,
            org_id=user.org_id,
            title=input_data.title
        )
        sql_db.add(ppt)
        sql_db.commit()
        sql_db.refresh(ppt)
        
        return f"✅ Created new PPT: '{ppt.title}' (ID: {ppt.id}) for this conversation. You can now add slides."
    except Exception as e:
        sql_db.rollback()
        return f"❌ Error creating PPT in DB: {str(e)}"

async def add_slide_db(input_data: AddSlideInput, sql_db: Session, conversation_id: str) -> str:
    """Add a slide to the current PowerPoint in the database"""
    try:
        # Find the latest presentation for this conversation
        ppt = sql_db.query(Presentation).filter(Presentation.conversation_id == conversation_id).order_by(Presentation.created_at.desc()).first()
        if not ppt:
            return "❌ No presentation found for this conversation. Please create one first using 'create_ppt'."

        # Get current slide count for ordering
        slide_count = sql_db.query(PresentationSlide).filter(PresentationSlide.presentation_id == ppt.id).count()
        
        slide = PresentationSlide(
            presentation_id=ppt.id,
            title=input_data.title,
            content=input_data.content,
            slide_type=input_data.slide_type,
            order=slide_count + 1
        )
        sql_db.add(slide)
        sql_db.commit()
        
        return f"✅ Added slide '{input_data.title}' to PPT '{ppt.title}'. Total slides: {slide_count + 1}"
    except Exception as e:
        sql_db.rollback()
        return f"❌ Error adding slide to DB: {str(e)}"

async def update_ppt_db(input_data: UpdatePPTInput, sql_db: Session, conversation_id: str) -> str:
    """Update the current PowerPoint presentation in the database"""
    try:
        ppt = sql_db.query(Presentation).filter(Presentation.conversation_id == conversation_id).order_by(Presentation.created_at.desc()).first()
        if not ppt:
            return "❌ No presentation found to update."

        if input_data.title:
            ppt.title = input_data.title
        
        if input_data.slides:
            # Delete existing slides and replace
            sql_db.query(PresentationSlide).filter(PresentationSlide.presentation_id == ppt.id).delete()
            for i, s_data in enumerate(input_data.slides):
                slide = PresentationSlide(
                    presentation_id=ppt.id,
                    title=s_data.get('title', ''),
                    content=s_data.get('content', ''),
                    slide_type=s_data.get('slide_type', 'bullet'),
                    order=i + 1
                )
                sql_db.add(slide)
        
        sql_db.commit()
        return f"✅ Updated PPT '{ppt.title}' successfully."
    except Exception as e:
        sql_db.rollback()
        return f"❌ Error updating PPT: {str(e)}"

def list_slides_db(sql_db: Session, conversation_id: str) -> str:
    """List slides of the current PPT in the database"""
    try:
        ppt = sql_db.query(Presentation).filter(Presentation.conversation_id == conversation_id).order_by(Presentation.created_at.desc()).first()
        if not ppt:
            return "📋 No presentation found for this conversation."
        
        slides = sql_db.query(PresentationSlide).filter(PresentationSlide.presentation_id == ppt.id).order_by(PresentationSlide.order.asc()).all()
        if not slides:
            return f"📋 PPT '{ppt.title}' has no slides yet."
        
        result = f"📋 **PPT: {ppt.title}** (ID: {ppt.id}, {len(slides)} slides)\n\n"
        for slide in slides:
            result += f"{slide.order}. **{slide.title}** ({slide.slide_type})\n"
        return result
    except Exception as e:
        return f"❌ Error retrieving slides: {str(e)}"

# --- Integration for Project (StructuredTool) ---

def ppt_generation_tool(sql_db: Session, user_id: str, conversation_id: Optional[str] = None) -> List[StructuredTool]:
    """Returns tools in StructuredTool format for project integration"""
    
    if not conversation_id:
        # Fallback if no conversation_id is provided, though it's required for DB storage
        return []

    async def create_ppt(title: str) -> str:
        """Initialize a new PowerPoint presentation state in the database."""
        return await create_ppt_db(CreatePPTInput(title=title), sql_db, user_id, conversation_id)

    async def add_slide(title: str, content: str, slide_type: str = "bullet") -> str:
        """Add a slide to the current PowerPoint presentation in the database."""
        return await add_slide_db(AddSlideInput(title=title, content=content, slide_type=slide_type), sql_db, conversation_id)

    def list_slides() -> str:
        """List all slides currently in the database for the current PowerPoint draft."""
        return list_slides_db(sql_db, conversation_id)

    async def update_ppt(title: Optional[str] = None, slides: Optional[List[Dict[str, str]]] = None) -> str:
        """Update the existing PowerPoint in the database."""
        return await update_ppt_db(UpdatePPTInput(title=title, slides=slides), sql_db, conversation_id)

    # Note: generate_ppt as a separate step is no longer strictly needed if we save per slide,
    # but we can keep it as a 'finalizer' or 'preview' tool if needed.
    # For now, following the user's "store in db" requirement.
    
    return [
        StructuredTool.from_function(
            coroutine=create_ppt,
            name="create_ppt",
            description="Initialize a new PowerPoint presentation record in the database. Use this first.",
            args_schema=CreatePPTInput
        ),
        StructuredTool.from_function(
            coroutine=add_slide,
            name="add_slide",
            description="Add a slide to the current database-stored PowerPoint presentation.",
            args_schema=AddSlideInput
        ),
        StructuredTool.from_function(
            func=list_slides,
            name="list_slides",
            description="List all slides stored in the database for the current presentation.",
            args_schema=None
        ),
        StructuredTool.from_function(
            coroutine=update_ppt,
            name="update_ppt",
            description="Update the current PowerPoint presentation in the database (title or slides).",
            args_schema=UpdatePPTInput
        )
    ]
