import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.infrastructure.graph.neo4j_client import get_neo4j_client
from src.infrastructure.graph.schema import ALLOWED_NODE_TYPES, ALLOWED_RELATIONSHIPS
from src.config.settings import settings

def inspect_neo4j():
    print("--- Neo4j Inspection Script ---")
    
    print(f"\n[Configuration]")
    print(f"Neo4j URI: {settings.neo4j_uri}")
    
    print("\n[Codebase Schema Definition]")
    print("Allowed Node Labels:")
    for label in sorted(ALLOWED_NODE_TYPES):
        print(f" - {label}")
        
    print("\nAllowed Relationship Types:")
    for rel in sorted(ALLOWED_RELATIONSHIPS):
        print(f" - {rel}")

    print("\n[Database Connection Check]")
    try:
        client = get_neo4j_client()
        # Verify connection
        client.verify()
        print("✅ Connection Successful.")
        
        print("\n[Database Stats]")
        print("--- Node Counts by Label ---")
        query_nodes = """
        MATCH (n)
        RETURN labels(n) as Labels, count(*) as Count
        ORDER BY Count DESC
        """
        nodes = client.run(query_nodes)
        if not nodes:
            print("No nodes found.")
        for record in nodes:
            print(f"{record['Labels']}: {record['Count']}")
            
        print("\n--- Relationship Counts by Type ---")
        query_rels = """
        MATCH ()-[r]->()
        RETURN type(r) as Type, count(*) as Count
        ORDER BY Count DESC
        """
        rels = client.run(query_rels)
        if not rels:
            print("No relationships found.")
        for record in rels:
            print(f"{record['Type']}: {record['Count']}")

        print("\n--- Relationship Schema (Source -> Rel -> Target) ---")
        query_schema = """
        MATCH (a)-[r]->(b)
        RETURN labels(a) as Source, type(r) as Relationship, labels(b) as Target, count(*) as Count
        ORDER BY Relationship, Count DESC
        """
        schema = client.run(query_schema)
        if not schema:
            print("No schema patterns found.")
        for record in schema:
            print(f"{record['Source']} -[{record['Relationship']}]-> {record['Target']} (Count: {record['Count']})")
            
        print("\n--- Sample Nodes (Limit 5) ---")
        query_sample = "MATCH (n) RETURN labels(n) as Labels, n LIMIT 5"
        samples = client.run(query_sample)
        for record in samples:
            print(f"Labels: {record['Labels']}, Properties: {json.dumps(record['n'], default=str)}")

        client.close()
        
    except Exception as e:
        print("❌ Connection Failed.")
        print(f"Error: {e}")
        print("\nCannot retrieve live data from the database. Please check your NEO4J_URI and credentials in .env")

if __name__ == "__main__":
    inspect_neo4j()
