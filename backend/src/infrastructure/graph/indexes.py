from src.infrastructure.graph.neo4j_client import get_neo4j_client
from src.infrastructure.graph.schema import EXTRACTABLE_NODE_TYPES

def create_indexes() -> None:
    client = get_neo4j_client()
    with client.session() as session:
        # System nodes
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:DocumentVersion) REQUIRE v.version_id IS UNIQUE"
        )
        
        # Domain nodes - Constraint on normalized_name for resolution
        for label in EXTRACTABLE_NODE_TYPES:
            # Create constraint for uniqueness/merging
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.normalized_name IS UNIQUE"
            )
            # Create index for display name search
            session.run(
                f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)"
            )
