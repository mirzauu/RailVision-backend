from typing import Optional, List
from sqlalchemy.orm import Session, joinedload
from src.infrastructure.database.models import Document

class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, document_id: str) -> Optional[Document]:
        return self.db.query(Document).options(joinedload(Document.uploader)).filter(Document.id == document_id).first()

    def get_by_org(self, org_id: str) -> List[Document]:
        return (
            self.db.query(Document)
            .options(joinedload(Document.uploader))
            .filter(Document.org_id == org_id)
            .filter(Document.project_id.is_(None))
            .order_by(Document.created_at.desc())
            .all()
        )

    def create(self, document: Document) -> Document:
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def update(self, document: Document) -> Document:
        self.db.commit()
        self.db.refresh(document)
        return document
