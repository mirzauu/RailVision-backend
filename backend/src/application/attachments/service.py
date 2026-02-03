"""
Attachment Service - Handles file attachment processing for chat.
Uses Pinecone only (no Neo4j) for RAG retrieval.
"""
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Optional
from src.infrastructure.ingestion.loader import load_document
from src.infrastructure.ingestion.segmenter import segment_pages
from src.infrastructure.vector.writer import upsert_segments_batch
from src.infrastructure.vector.retriever import retrieve_context

from src.infrastructure.database.models import Document, DocumentStatus, DocumentType, DocumentScope
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class AttachmentService:
    """Service for processing chat attachments and retrieving context."""
    
    STORAGE_ROOT = Path("storage") / "attachments"
    
    def __init__(self):
        self.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

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
    
    def _sanitize_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        # Remove NULL bytes which PostgreSQL doesn't support
        return text.replace("\x00", "")

    async def process_attachment(
        self,
        db: Session,
        file_bytes: bytes,
        filename: str,
        user_id: str,
        org_id: str,
        project_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> str:
        """
        Process an uploaded attachment file.
        
        1. Save file to disk
        2. Extract text using loader
        3. Segment into chunks
        4. Index to Pinecone
        5. Save record to PostgreSQL
        
        Returns:
            attachment_id: Unique ID for this attachment (used for retrieval)
        """
        attachment_id = str(uuid.uuid4())
        
        # Save file to storage
        safe_filename = f"{attachment_id}_{filename}"
        file_path = self.STORAGE_ROOT / user_id
        file_path.mkdir(parents=True, exist_ok=True)
        target_path = file_path / safe_filename
        target_path.write_bytes(file_bytes)
        
        logger.info(f"Saved attachment to {target_path}")
        
        # Create Document record in DB
        doc = Document(
            id=attachment_id,
            org_id=org_id,
            project_id=project_id,
            uploaded_by=user_id,
            filename=safe_filename,
            original_filename=filename,
            file_type=self._infer_type(filename),
            file_size_bytes=len(file_bytes),
            storage_path=str(target_path),
            storage_backend="local",
            status=DocumentStatus.UPLOADING,
            scope=DocumentScope.PRIVATE, # Attachments are private to the conversation
            metadata_={"message_id": message_id} if message_id else {}
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        try:
            # Extract text from document
            pages = load_document(target_path)
            logger.info(f"Extracted {len(pages)} pages from attachment")
            
            # Sanitize extracted text
            for page in pages:
                if "text" in page:
                    page["text"] = self._sanitize_text(page["text"])

            if not pages:
                logger.warning(f"No content extracted from {filename}")
                doc.status = DocumentStatus.FAILED
                doc.ingestion_error = "No content extracted"
                db.commit()
                return attachment_id
            
            doc.extracted_text = "\n".join([p.get("text", "") for p in pages])
            doc.text_length = len(doc.extracted_text)
            doc.page_count = len(pages)
            doc.status = DocumentStatus.PROCESSING
            db.commit()

            # Segment pages into chunks
            segments = segment_pages(pages)
            logger.info(f"Created {len(segments)} segments from attachment")
            
            # Prepare segments for Pinecone
            pinecone_segments = []
            for i, seg in enumerate(segments):
                pinecone_segments.append({
                    "doc_id": f"attachment_{attachment_id}",
                    "doc_version": "1",
                    "segment_id": f"chunk_{i}",
                    "text": seg.get("text", ""),
                    "category": "attachment",
                    "page_numbers": seg.get("page_numbers", []),
                    "classification_confidence": 1.0,
                })
            
            # Index to Pinecone
            if pinecone_segments:
                upsert_segments_batch(pinecone_segments)
                logger.info(f"Indexed {len(pinecone_segments)} chunks to Pinecone")
                doc.status = DocumentStatus.INGESTED
                doc.chunks_count = len(pinecone_segments)
                db.commit()
            
            return attachment_id
            
        except Exception as e:
            # Important: rollback if commit failed
            db.rollback()
            logger.error(f"Failed to process attachment {filename}: {e}", exc_info=True)
            doc.status = DocumentStatus.FAILED
            doc.ingestion_error = self._sanitize_text(str(e))
            db.commit()
            raise
    
    def retrieve_attachment_context(
        self,
        query: str,
        attachment_id: str,
        top_k: int = 5,
    ) -> str:
        """
        Retrieve relevant context from an attachment using RAG.
        
        Args:
            query: The user's question
            attachment_id: ID of the attachment to search in
            top_k: Number of chunks to retrieve
            
        Returns:
            Concatenated text from the most relevant chunks
        """
        try:
            doc_id = f"attachment_{attachment_id}"
            results = retrieve_context(
                query=query,
                doc_id=doc_id,
                top_k=top_k,
            )
            
            if not results:
                logger.info(f"No context found for query in attachment {attachment_id}")
                return ""
            
            # Extract text from results
            context_parts = []
            for result in results:
                metadata = result.get("text", {})
                if isinstance(metadata, dict):
                    text = metadata.get("text", "")
                else:
                    text = str(metadata)
                if text:
                    context_parts.append(text)
            
            context = "\n\n---\n\n".join(context_parts)
            logger.info(f"Retrieved {len(context_parts)} chunks for context ({len(context)} chars)")
            return self._sanitize_text(context)
            
        except Exception as e:
            logger.error(f"Failed to retrieve context for attachment {attachment_id}: {e}")
            return ""
    
    def retrieve_context_for_attachments(
        self,
        query: str,
        attachment_ids: List[str],
        top_k: int = 5,
    ) -> str:
        """
        Retrieve context from multiple attachments.
        """
        all_context = []
        for att_id in attachment_ids:
            ctx = self.retrieve_attachment_context(query, att_id, top_k=top_k)
            if ctx:
                all_context.append(ctx)
        
        return "\n\n===\n\n".join(all_context)
