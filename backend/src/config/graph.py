from typing import Generator
from neo4j import Session
from src.infrastructure.graph.neo4j_client import get_neo4j_client

def get_graph_session() -> Generator[Session, None, None]:
    client = get_neo4j_client()
    s = client.session()
    try:
        yield s
    finally:
        s.close()

