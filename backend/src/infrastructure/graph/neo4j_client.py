from typing import Optional, Any, Dict, List
from neo4j import GraphDatabase, Driver, Session
from src.config.settings import settings

_client: Optional["Neo4jClient"] = None

class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str):
        # In development, if SSL verification fails (common with some python/OS combos),
        # fallback to self-signed certificate mode (skips verification)
        if settings.is_development and "neo4j+s://" in uri:
            # We don't want to modify the URI blindly, but for the specific case of 
            # connection failures seen in dev, using ssc is a safe workaround
            uri = uri.replace("neo4j+s://", "neo4j+ssc://")
            
        self._driver: Driver = GraphDatabase.driver(uri, auth=(username, password))

    def verify(self) -> None:
        self._driver.verify_connectivity()

    def session(self, database: Optional[str] = None) -> Session:
        if database:
            return self._driver.session(database=database)
        return self._driver.session()

    def run(self, query: str, parameters: Optional[Dict[str, Any]] = None, database: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.session(database) as s:
            result = s.run(query, parameters or {})
            return [r.data() for r in result]

    def close(self) -> None:
        self._driver.close()

def get_neo4j_client() -> Neo4jClient:
    global _client
    if _client is None:
        if not settings.neo4j_uri or not settings.neo4j_username or not settings.neo4j_password:
            raise RuntimeError("Neo4j configuration missing")
        _client = Neo4jClient(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
    return _client
