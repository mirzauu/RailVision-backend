import sys
import os
import time

# Add the backend directory to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from src.infrastructure.vector.writer import upsert_segment
    from src.infrastructure.vector.retriever import retrieve_context
    from src.config.settings import settings
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def run_verification():
    if not settings.pinecone_api_key or not settings.openai_api_key:
        print("Error: PINECONE_API_KEY or OPENAI_API_KEY not set in environment/settings.")
        return

    print(f"Testing Pinecone connection... Index: {settings.pinecone_index_name}")
    
    # Test Data
    doc_id = "railvision_pitchdeck"
    doc_version = "railvision_pitchdeck:ab12cd34"
    category = "go_to_market"
    segment_id = f"{doc_id}:{doc_version}:test_seg_01"
    text = "We plan to go to market with an aggressive pricing strategy focused on enterprise clients."
    
    print("\n--- Step 1: Upserting Test Segment ---")
    try:
        upsert_segment(
            segment_id=segment_id,
            text=text,
            metadata={
                "doc_id": doc_id,
                "doc_version": doc_version,
                "category": category,
                "page_numbers": [14],
                "confidence": 0.95
            }
        )
        print("Upsert successful.")
    except Exception as e:
        print(f"Upsert failed: {e}")
        return

    print("\n--- Waiting for consistency (5s) ---")
    time.sleep(5)

    print("\n--- Step 2: Retrieving Context ---")
    try:
        results = retrieve_context(
            query="go to market strategy",
            doc_id=doc_id,
            active_version=doc_version,
            allowed_categories=[category]
        )
        
        if results:
            print(f"Found {len(results)} matches.")
            for i, res in enumerate(results):
                # The retriever returns dicts with 'text' (which is metadata) and 'score'
                metadata = res['text']
                score = res['score']
                print(f"Match {i+1} (Score: {score:.4f})")
                print(f"  Metadata: {metadata}")
        else:
            print("No matches found. (Is the index fresh?)")
            
    except Exception as e:
        print(f"Retrieval failed: {e}")

if __name__ == "__main__":
    run_verification()
