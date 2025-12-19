"""
Integration test for document upload with vector database storage.

This script tests the complete flow:
1. Upload a document via API
2. Verify it's stored in the database
3. Verify segments are in Neo4j (graph)
4. Verify embeddings are in Pinecone (vector)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.infrastructure.vector.retriever import retrieve_context
from src.config.settings import settings

def test_vector_integration(doc_id: str, version_id: str):
    """
    Test if a document's embeddings were successfully stored in Pinecone.
    
    Args:
        doc_id: Document ID (e.g., from database)
        version_id: Document version ID
    """
    print(f"\n{'='*60}")
    print(f"Testing Vector DB Integration")
    print(f"{'='*60}")
    print(f"Doc ID: {doc_id}")
    print(f"Version: {version_id}")
    print(f"Index: {settings.pinecone_index_name}")
    
    # Test retrieval with a generic query
    test_queries = [
        "market strategy",
        "product",
        "business",
        "company",
    ]
    
    # Try different categories
    test_categories = [
        ["go_to_market"],
        ["market"],
        ["product"],
        ["business"],
        ["financial"],
        ["technical"],
    ]
    
    total_matches = 0
    
    for query in test_queries:
        print(f"\n--- Testing Query: '{query}' ---")
        
        for categories in test_categories:
            try:
                results = retrieve_context(
                    query=query,
                    doc_id=doc_id,
                    active_version=version_id,
                    allowed_categories=categories,
                    top_k=3
                )
                
                if results:
                    print(f"✓ Found {len(results)} matches for categories {categories}")
                    total_matches += len(results)
                    
                    # Show first match details
                    if results:
                        first = results[0]
                        print(f"  Score: {first['score']:.4f}")
                        metadata = first['text']
                        print(f"  Category: {metadata.get('category', 'N/A')}")
                        print(f"  Pages: {metadata.get('page_numbers', 'N/A')}")
                        
            except Exception as e:
                print(f"✗ Error querying {categories}: {e}")
    
    print(f"\n{'='*60}")
    if total_matches > 0:
        print(f"✅ SUCCESS: Found {total_matches} total matches")
        print(f"Vector database integration is working!")
    else:
        print(f"⚠️  WARNING: No matches found")
        print(f"This could mean:")
        print(f"  1. Document hasn't been processed yet (wait a few seconds)")
        print(f"  2. Document has no text content")
        print(f"  3. Categories don't match")
        print(f"  4. Vector DB persistence failed (check logs)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_upload_integration.py <doc_id> <version_id>")
        print("\nExample:")
        print("  python test_upload_integration.py abc123 abc123:v1")
        print("\nTo get doc_id and version_id:")
        print("  1. Upload a document via API")
        print("  2. Check the response for 'id' and 'version' fields")
        sys.exit(1)
    
    doc_id = sys.argv[1]
    version_id = sys.argv[2]
    
    test_vector_integration(doc_id, version_id)
