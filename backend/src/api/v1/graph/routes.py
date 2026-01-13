from fastapi import APIRouter, UploadFile, File, Form, Depends
from uuid import uuid4
from pathlib import Path
from typing import Optional, Dict, Any
from src.api.dependencies import get_current_user
from src.infrastructure.database.models import User
from src.application.graph.service import GraphService
from src.infrastructure.ingestion.pipeline import run_ingestion
from src.infrastructure.graph.neo4j_client import get_neo4j_client
from .schemas import (
    TestIngestionGraphResponse, 
    IngestionSummary, 
    GraphSummary, 
    GraphPreviewNode, 
    GraphPreviewRel,
    GraphVisualizationResponse,
    GraphNode,
    GraphRelationship
)

router = APIRouter()

@router.post("/test", response_model=TestIngestionGraphResponse)
async def test_ingestion_and_graph(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    persist: bool = Form(True),
    current_user: User = Depends(get_current_user),
):
    # Allow common document types: PDF, DOCX, TXT, MD
    allowed_exts = (".pdf", ".docx", ".txt", ".md")
    orig_ext = Path(file.filename).suffix.lower()
    
    if orig_ext not in allowed_exts and file.content_type != "application/octet-stream":
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported extensions: {', '.join(allowed_exts)}"
        )

    tmp_root = Path("storage") / "test"
    tmp_root.mkdir(parents=True, exist_ok=True)
    doc_id = str(uuid4())
    version_id = "1"
    
    # Preserve original extension
    save_ext = orig_ext if orig_ext in allowed_exts else (".pdf" if file.content_type == "application/pdf" else ".txt")
    save_path = tmp_root / f"{doc_id}{save_ext}"

    data = await file.read()
    if not data:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
    with open(save_path, "wb") as f:
        f.write(data)
        f.flush()

    # Use GraphService for unified processing
    svc = GraphService()
    
    if persist:
        title_ = title or file.filename
        segments = await svc.ingest_and_persist(
            file_path=save_path,
            doc_id=doc_id,
            version_id=version_id,
            title=title_,
            doc_type=save_ext.lstrip('.') or "pdf",
            content_hash=None, # Service will calculate it
        )
    else:
        # Fallback to direct pipeline run if persist=False
        segments = await run_ingestion(save_path, doc_id=doc_id, version_id=version_id)

    entity_count = sum(len(s.get("entities", [])) for s in segments)
    rel_count = sum(len(s.get("relationships", [])) for s in segments)

    graph_summary: Optional[GraphSummary] = None
    if persist:
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

@router.get("/visualization", response_model=GraphVisualizationResponse)
async def get_graph_visualization(
    limit: int = 1000,
    current_user: User = Depends(get_current_user),
):
    """
    Get graph data for visualization.
    Returns nodes and relationships up to the specified limit.
    """
    client = get_neo4j_client()
    query = """
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN n, r, m
    LIMIT $limit
    """
    
    with client.session() as session:
        result = session.run(query, limit=limit)
        
        nodes_map = {}
        links_map = {}
        
        for record in result:
            n = record["n"]
            r = record["r"]
            m = record["m"]
            
            # Process node n
            if n is not None:
                # Use element_id if available (Neo4j 5+), else fallback to id
                n_id = getattr(n, "element_id", str(n.id))
                if n_id not in nodes_map:
                    nodes_map[n_id] = GraphNode(
                        id=n_id,
                        labels=list(n.labels),
                        properties=dict(n)
                    )
            
            # Process node m
            if m is not None:
                m_id = getattr(m, "element_id", str(m.id))
                if m_id not in nodes_map:
                    nodes_map[m_id] = GraphNode(
                        id=m_id,
                        labels=list(m.labels),
                        properties=dict(m)
                    )
            
            # Process relationship r
            if r is not None:
                r_id = getattr(r, "element_id", str(r.id))
                if r_id not in links_map:
                    # Resolve start/end node IDs
                    start_node_id = getattr(r.start_node, "element_id", str(r.start_node.id))
                    end_node_id = getattr(r.end_node, "element_id", str(r.end_node.id))
                    
                    links_map[r_id] = GraphRelationship(
                        id=r_id,
                        source=start_node_id,
                        target=end_node_id,
                        type=r.type,
                        properties=dict(r)
                    )
                    
        return GraphVisualizationResponse(
            nodes=list(nodes_map.values()),
            links=list(links_map.values())
        )
