from src.infrastructure.graph.neo4j_client import get_neo4j_client

def create_indexes() -> None:
    client = get_neo4j_client()
    with client.session() as session:
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:DocumentVersion) REQUIRE v.version_id IS UNIQUE"
        )

