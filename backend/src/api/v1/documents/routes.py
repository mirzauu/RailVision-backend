from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from src.config.database import get_db
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User
from src.infrastructure.database.repositories.document_repository import DocumentRepository
from src.application.documents.service import DocumentService
from src.api.v1.documents.schemas import DocumentResponse

router = APIRouter()

def get_document_service(db: Session = Depends(get_db)) -> DocumentService:
    return DocumentService(DocumentRepository(db))

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    scope: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    svc: DocumentService = Depends(get_document_service),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not current_user.org_id:
        raise Exception("Organization not found")
    data = await file.read()
    tag_list = []
    if tags:
        try:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        except Exception:
            tag_list = []
    doc = await svc.upload(
        db=db,
        org_id=current_user.org_id,
        user_id=current_user.id,
        original_filename=file.filename,
        file_bytes=data,
        mime_type=file.content_type,
        title=title,
        description=description,
        scope=scope,
        assigned_agent_ids=None,
        category=category,
        tags=tag_list,
    )
    return doc

@router.get("/", response_model=List[DocumentResponse])
def list_documents(
    svc: DocumentService = Depends(get_document_service),
    current_user: User = Depends(get_current_user),
):
    if not current_user.org_id:
        return []
    return svc.list_by_org(current_user.org_id)

@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    svc: DocumentService = Depends(get_document_service),
    current_user: User = Depends(get_current_user),
):
    d = svc.get(document_id)
    if not d or d.org_id != current_user.org_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Document not found")
    return d
