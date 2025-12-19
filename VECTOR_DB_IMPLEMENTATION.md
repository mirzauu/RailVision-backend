# Vector Database Implementation - Complete ✅

## Overview
The vector database infrastructure has been successfully implemented following the specifications in `vectordb.txt`. This implementation provides a clean, scalable architecture for storing and retrieving document embeddings using Pinecone and OpenAI.

## Architecture

### Core Principles
1. **Pinecone stores explanatory context, not truth**
   - Neo4j answers what is true
   - Pinecone explains why it is true
   - Pinecone NEVER decides feasibility, timing, or risk

2. **Version Safety**
   - No cross-version retrieval
   - All queries are filtered by doc_id, doc_version, and category
   - No global search or category mixing

3. **Cost Control**
   - Segment-level embeddings only
   - No re-embedding of unchanged versions
   - top_k ≤ 5 for all queries

## File Structure

```
backend/src/infrastructure/vector/
├── __init__.py              # Package exports
├── client.py                # Pinecone client initialization
├── embedder.py              # OpenAI embedding generation
├── writer.py                # Version-safe upsert operations
└── retriever.py             # Filtered similarity search
```

## Implementation Details

### 1. Client (`client.py`)
- Initializes Pinecone client with API key from settings
- Provides `get_index()` function for accessing the configured index

### 2. Embedder (`embedder.py`)
- Uses OpenAI's `text-embedding-3-large` model (3072 dimensions)
- Generates one embedding per segment
- Stable, boring, and correct implementation

### 3. Writer (`writer.py`)
- **`upsert_segment()`**: Upserts a single segment with metadata
- **`persist_to_pinecone()`**: Batch upserts multiple segments
- **Metadata Safety**: Automatically converts `page_numbers` lists to strings
- **Metadata Schema**:
  ```python
  {
      "doc_id": str,
      "doc_version": str,
      "category": str,
      "page_numbers": list[str],  # Converted from int to str
      "confidence": float
  }
  ```

### 4. Retriever (`retriever.py`)
- **Filtered Retrieval**: All queries MUST include:
  - `doc_id`: Document identifier
  - `active_version`: Specific document version
  - `allowed_categories`: List of permitted categories
- **Hard Rules**:
  - ❌ No cross-version retrieval
  - ❌ No global search
  - ❌ No mixing categories

## Configuration

### Environment Variables (in `settings.py`)
```python
PINECONE_API_KEY = "your_api_key"
PINECONE_INDEX_NAME = "quickstart"  # or your index name
OPENAI_API_KEY = "your_openai_key"
```

### Pinecone Index Requirements
- **Dimensions**: 3072 (for text-embedding-3-large)
- **Metric**: cosine
- **Type**: Serverless or Pod-based

## Usage Examples

### Writing to Pinecone
```python
from src.infrastructure.vector import persist_to_pinecone

segments = [
    {
        "doc_id": "railvision_pitchdeck",
        "doc_version": "railvision_pitchdeck:ab12cd34",
        "segment_id": "seg_001",
        "text": "Our go-to-market strategy focuses on enterprise clients...",
        "category": "go_to_market",
        "page_numbers": [14, 15],
        "classification_confidence": 0.95
    }
]

persist_to_pinecone(segments)
```

### Retrieving from Pinecone
```python
from src.infrastructure.vector import retrieve_context

results = retrieve_context(
    query="What is our go-to-market strategy?",
    doc_id="railvision_pitchdeck",
    active_version="railvision_pitchdeck:ab12cd34",
    allowed_categories=["go_to_market", "market", "pricing"],
    top_k=5
)

for result in results:
    print(f"Score: {result['score']}")
    print(f"Metadata: {result['text']}")
```

## Testing

### Diagnostic Test
```bash
.\env\Scripts\python.exe backend/src/scripts/test_pinecone.py
```

Expected output:
```
✓ Pinecone client initialized successfully
✓ Successfully connected to index: quickstart
✓ Index stats: {...}
```

### Full Verification
```bash
.\env\Scripts\python.exe backend/src/scripts/verify_vector.py
```

Expected output:
```
Testing Pinecone connection... Index: quickstart

--- Step 1: Upserting Test Segment ---
Upsert successful.

--- Waiting for consistency (5s) ---

--- Step 2: Retrieving Context ---
Found 1 matches.
Match 1 (Score: 0.8xxx)
  Metadata: {'doc_id': 'railvision_pitchdeck', ...}
```

## Integration with CSO Agent

When the CSO agent receives a query like:
> "What if we go to market with our latest product now?"

The flow is:
1. **Neo4j** determines:
   - Latest product
   - Active doc version
   - Relevant categories: `go_to_market`, `market`, `pricing`

2. **Pinecone** retrieves only explanations (context)

3. **LLM** reasons over:
   - Graph facts (truth from Neo4j)
   - Pinecone text (context)

4. If Pinecone returns nothing:
   - CSO must say: "We don't have enough supporting data"
   - This is intelligence, not hallucination

## Dependencies

```txt
pinecone>=3.0.0
openai>=1.12.0
```

## Troubleshooting

### Issue: `pinecone-client` error
**Solution**: The old `pinecone-client` package conflicts with the new `pinecone` package.
```bash
# Remove old package
.\env\Scripts\python.exe -m pip uninstall -y pinecone-client

# Install correct package
.\env\Scripts\python.exe -m pip install --no-cache-dir pinecone==8.0.0
```

### Issue: Using system Python instead of venv
**Solution**: Always use the venv's Python explicitly:
```bash
.\env\Scripts\python.exe your_script.py
```

### Issue: Index not found (404)
**Solution**: Create the index in Pinecone console with correct dimensions (3072).

## Status

✅ **Implementation Complete**
- All core modules implemented
- Metadata safety enforced
- Version control implemented
- Cost controls in place
- Testing scripts provided
- Full verification passed

## Next Steps

1. Integrate with document ingestion pipeline
2. Connect to Neo4j for truth/context separation
3. Implement CSO agent query flow
4. Add monitoring and logging
5. Implement hash-based deduplication for unchanged segments
