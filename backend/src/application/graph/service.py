from pathlib import Path
from hashlib import sha256
from typing import List, Dict, Optional
from src.infrastructure.ingestion.pipeline import run_ingestion
from src.infrastructure.graph.writer import persist_to_graph

class GraphService:
    async def ingest_and_persist(
        self,
        file_path: Path,
        doc_id: str,
        version_id: str,
        title: str,
        doc_type: str,
        content_hash: Optional[str] = None,
    ) -> None:
        processed: List[Dict] = await run_ingestion(file_path, doc_id, version_id)
        h = content_hash
        if not h:
            data = Path(file_path).read_bytes()
            h = sha256(data).hexdigest()
        persist_to_graph(
            processed_segments=processed,
            doc_id=doc_id,
            version_id=version_id,
            hash=h,
            title=title,
            doc_type=doc_type,
        )

