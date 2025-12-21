import sys
import os

# Add backend to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.application.reasoning.state_builder import build_state
from src.application.reasoning.pipeline import context_enrich

def main():
    print("Verifying Reasoning Module (Updated)...")
    
    query = "investment"
    print(f"\n1. Testing 'context_enrich' with query: '{query}'")
    
    # This simulates the full pipeline
    # It will try to hit Pinecone (might fail if no creds/data) and then Neo4j
    try:
        enriched = context_enrich(query)
        print("Enrichment successful.")
        print("--- PREFIX OF ENRICHED PROMPT ---")
        print(enriched[:500] + "...")
        print("---------------------------------")
    except Exception as e:
        print(f"Context Enrich execution failed: {e}")

    print(f"\n2. Testing 'build_state' directly with Keyword Filter: '{query}'")
    try:
        state = build_state(query_text=query)
        print(f"Items found matching '{query}': {len(state)}")
        for item in state[:3]: # Show top 3
            print(item)
    except Exception as e:
        print(f"State build failed: {e}")

if __name__ == "__main__":
    main()
