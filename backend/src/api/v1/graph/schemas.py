from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class IngestionSummary(BaseModel):
    pages: int
    segments: int
    entities: int
    relationships: int

class GraphPreviewRel(BaseModel):
    type: str
    from_name: Optional[str] = None
    to_name: Optional[str] = None

class GraphPreviewNode(BaseModel):
    labels: List[str]
    name: Optional[str] = None

class GraphSummary(BaseModel):
    entity_count: int
    relationship_count: int
    version_count: int
    sample_nodes: List[GraphPreviewNode]
    sample_relationships: List[GraphPreviewRel]

class TestIngestionGraphResponse(BaseModel):
    doc_id: str
    version_id: str
    file_saved_at: str
    ingestion: IngestionSummary
    graph: Optional[GraphSummary] = None

class GraphNode(BaseModel):
    id: str
    labels: List[str]
    properties: Dict[str, Any]

class GraphRelationship(BaseModel):
    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any]

class GraphVisualizationResponse(BaseModel):
    nodes: List[GraphNode]
    links: List[GraphRelationship]

