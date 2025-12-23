from pathlib import Path
from hashlib import sha256
from typing import List, Dict, Optional
import logging
from src.infrastructure.ingestion.pipeline import run_ingestion
from src.infrastructure.graph.writer import persist_to_graph
from src.infrastructure.vector.writer import persist_to_pinecone

logger = logging.getLogger(__name__)

class GraphService:
    async def ingest_and_persist(
        self,
        file_path: Path,
        doc_id: str,
        version_id: str,
        title: str,
        doc_type: str,
        content_hash: Optional[str] = None,
    ) -> List[Dict]:
        
        logger.info(f"Starting ingestion for doc_id={doc_id}, version={version_id}")
        
        # Run ingestion pipeline
        processed: List[Dict] = await run_ingestion(file_path, doc_id, version_id)
        logger.info(f"Ingestion complete: {len(processed)} segments processed")
        
        # Calculate content hash if not provided
        h = content_hash
        if not h:
            data = Path(file_path).read_bytes()
            h = sha256(data).hexdigest()
        
        # Persist to Neo4j (truth - what is factually correct)
        logger.info(f"Persisting {len(processed)} segments to Neo4j (graph DB)")
        persist_to_graph(
            processed_segments=processed,
            doc_id=doc_id,
            version_id=version_id,
            hash=h,
            title=title,
            doc_type=doc_type,
        )
        logger.info("Neo4j persistence complete")
        
        # Persist to Pinecone (context - explanatory information)
        # Only persist segments with text content
        vector_segments = [seg for seg in processed if seg.get("text")]
        if vector_segments:
            try:
                logger.info(f"Persisting {len(vector_segments)} segments to Pinecone (vector DB)")
                persist_to_pinecone(vector_segments)
                logger.info("Pinecone persistence complete")
            except Exception as e:
                # Log error but don't fail the entire operation
                # Graph DB (truth) is more critical than vector DB (context)
                logger.error(f"Failed to persist to Pinecone: {e}", exc_info=True)
                logger.warning("Document uploaded successfully but vector embeddings were not created")
        else:
            logger.warning(f"No text segments found for vector storage (doc_id={doc_id})")
            
        return processed
