from pathlib import Path
from typing import Optional, List
from sqlalchemy.orm import Session
from src.infrastructure.database.models import Document, DocumentStatus, DocumentType, DocumentScope
from src.infrastructure.database.repositories.document_repository import DocumentRepository

class DocumentService:
    def __init__(self, repo: DocumentRepository):
        self.repo = repo

    def _infer_type(self, filename: str) -> DocumentType:
        ext = (filename.split(".")[-1] or "").lower()
        mapping = {
            "pdf": DocumentType.PDF,
            "docx": DocumentType.DOCX,
            "pptx": DocumentType.PPTX,
            "xlsx": DocumentType.XLSX,
            "csv": DocumentType.CSV,
            "txt": DocumentType.TXT,
            "md": DocumentType.MD,
            "json": DocumentType.JSON,
            "xml": DocumentType.XML,
        }
        return mapping.get(ext, DocumentType.TXT)

    def upload(
        self,
        db: Session,
        org_id: str,
        user_id: str,
        original_filename: str,
        file_bytes: bytes,
        mime_type: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        scope: Optional[str] = None,
        assigned_agent_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Document:
        storage_root = Path("storage") / "documents" / org_id
        storage_root.mkdir(parents=True, exist_ok=True)
        safe_name = original_filename
        target_path = storage_root / safe_name
        idx = 1
        while target_path.exists():
            stem = safe_name
            if "." in safe_name:
                stem = safe_name.rsplit(".", 1)[0]
                ext = safe_name.rsplit(".", 1)[1]
                safe_name = f"{stem}_{idx}.{ext}"
            else:
                safe_name = f"{stem}_{idx}"
            target_path = storage_root / safe_name
            idx += 1
        target_path.write_bytes(file_bytes)
        doc = Document(
            org_id=org_id,
            uploaded_by=user_id,
            filename=safe_name,
            original_filename=original_filename,
            file_type=self._infer_type(original_filename),
            mime_type=mime_type,
            file_size_bytes=len(file_bytes),
            storage_path=str(target_path),
            storage_backend="local",
            status=DocumentStatus.UPLOADED,
            title=title,
            description=description,
            scope=DocumentScope(scope) if scope else DocumentScope.ORGANIZATION,
            assigned_agent_ids=assigned_agent_ids or [],
            category=category,
            tags=tags or [],
        )
        return self.repo.create(doc)

    def list_by_org(self, org_id: str) -> List[Document]:
        return self.repo.get_by_org(org_id)

    def get(self, document_id: str) -> Optional[Document]:
        return self.repo.get_by_id(document_id)
