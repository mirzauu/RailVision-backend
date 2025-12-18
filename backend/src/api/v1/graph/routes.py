from fastapi import APIRouter, UploadFile, File, Form, Depends
from uuid import uuid4
from pathlib import Path
from typing import Optional, Dict, Any
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User
from src.infrastructure.ingestion.pipeline import run_ingestion
from src.infrastructure.graph.writer import persist_to_graph
from src.infrastructure.graph.neo4j_client import get_neo4j_client
from .schemas import TestIngestionGraphResponse, IngestionSummary, GraphSummary, GraphPreviewNode, GraphPreviewRel

router = APIRouter()

@router.post("/test", response_model=TestIngestionGraphResponse)
async def test_ingestion_and_graph(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    persist: bool = Form(True),
    current_user: User = Depends(get_current_user),
):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Only PDF files are supported for this test")

    tmp_root = Path("storage") / "test"
    tmp_root.mkdir(parents=True, exist_ok=True)
    doc_id = str(uuid4())
    version_id = "1"
    save_path = tmp_root / f"{doc_id}.pdf"
    data = await file.read()
    save_path.write_bytes(data)

    segments = await run_ingestion(save_path, doc_id=doc_id, version_id=version_id)
    entity_count = sum(len(s.get("entities", [])) for s in segments)
    rel_count = sum(len(s.get("relationships", [])) for s in segments)

    graph_summary: Optional[GraphSummary] = None
    if persist:
        title_ = title or file.filename
        from hashlib import sha256
        content_hash = sha256(data).hexdigest()
        persist_to_graph(
            processed_segments=segments,
            doc_id=doc_id,
            version_id=version_id,
            hash=content_hash,
            title=title_,
            doc_type="pdf",
        )
        client = get_neo4j_client()
        with client.session() as s:
            e_cnt = s.run("MATCH (n) WHERE n.source_doc_id = $doc_id RETURN count(n) as c", {"doc_id": doc_id}).single()["c"]
            r_cnt = s.run("MATCH ()-[r]->() WHERE r.source_doc_id = $doc_id RETURN count(r) as c", {"doc_id": doc_id}).single()["c"]
            v_cnt = s.run("MATCH (:Document {doc_id: $doc_id})-[:HAS_VERSION]->(v:DocumentVersion) RETURN count(v) as c", {"doc_id": doc_id}).single()["c"]
            nodes_res = s.run(
                "MATCH (n) WHERE n.source_doc_id = $doc_id RETURN labels(n) as labels, n.name as name LIMIT 10",
                {"doc_id": doc_id},
            )
            rels_res = s.run(
                "MATCH (a)-[r]->(b) WHERE r.source_doc_id = $doc_id RETURN type(r) as t, a.name as a, b.name as b LIMIT 10",
                {"doc_id": doc_id},
            )
            nodes_preview = []
            for row in nodes_res:
                d = row.data()
                nodes_preview.append(GraphPreviewNode(labels=d.get("labels", []), name=d.get("name")))
            rels_preview = []
            for row in rels_res:
                d = row.data()
                rels_preview.append(GraphPreviewRel(type=d.get("t"), from_name=d.get("a"), to_name=d.get("b")))
            graph_summary = GraphSummary(
                entity_count=e_cnt,
                relationship_count=r_cnt,
                version_count=v_cnt,
                sample_nodes=nodes_preview,
                sample_relationships=rels_preview,
            )

    return TestIngestionGraphResponse(
        doc_id=doc_id,
        version_id=version_id,
        file_saved_at=str(save_path),
        ingestion=IngestionSummary(
            pages=len({pn for seg in segments for pn in seg.get('page_numbers', [])}),
            segments=len(segments),
            entities=entity_count,
            relationships=rel_count,
        ),
        graph=graph_summary,
    )
