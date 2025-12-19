# Vector Database Upload Integration

## Overview

The vector database has been successfully integrated with the document upload API. When a document is uploaded, it is now automatically:

1. **Stored in the database** (PostgreSQL/SQLite)
2. **Ingested and processed** (segmented, classified, validated)
3. **Persisted to Neo4j** (graph database - stores factual truth)
4. **Persisted to Pinecone** (vector database - stores explanatory context)

## Architecture

### Data Flow

```
User Upload (API)
    ↓
DocumentService.upload()
    ↓
GraphService.ingest_and_persist()
    ↓
    ├─→ run_ingestion() → Processed Segments
    ↓
    ├─→ persist_to_graph() → Neo4j (Truth)
    ↓
    └─→ persist_to_pinecone() → Pinecone (Context)
```

### Key Principles

1. **Separation of Concerns**
   - **Neo4j (Graph DB)**: Stores factual relationships and truth
   - **Pinecone (Vector DB)**: Stores semantic embeddings for context retrieval

2. **Error Handling**
   - Graph DB persistence is **critical** - failures will fail the upload
   - Vector DB persistence is **optional** - failures are logged but don't break the upload
   - This ensures documents are always accessible even if embeddings fail

3. **Data Consistency**
   - Both stores use the same `doc_id` and `version_id`
   - Segments are filtered: only those with text content get embeddings
   - Metadata is automatically sanitized (lists converted to strings)

## Implementation Details

### Modified Files

#### 1. `src/application/graph/service.py`

Added vector database integration:

```python
from src.infrastructure.vector.writer import persist_to_pinecone

class GraphService:
    async def ingest_and_persist(self, ...):
        # ... existing code ...
        
        # Persist to Pinecone (context - explanatory information)
        vector_segments = [seg for seg in processed if seg.get("text")]
        if vector_segments:
            try:
                persist_to_pinecone(vector_segments)
            except Exception as e:
                logger.error(f"Failed to persist to Pinecone: {e}")
```

**Key Features:**
- ✅ Automatic filtering of segments with text
- ✅ Error handling to prevent upload failures
- ✅ Comprehensive logging for debugging
- ✅ Version-safe storage

### Data Structure

Each segment stored in Pinecone contains:

```python
{
    "id": "doc_id:version_id:segment_id",
    "values": [0.123, 0.456, ...],  # 3072-dim embedding
    "metadata": {
        "doc_id": "abc123",
        "doc_version": "abc123:v1",
        "category": "go_to_market",
        "page_numbers": ["14", "15"],  # Converted to strings
        "confidence": 0.95
    }
}
```

## Usage

### 1. Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf" \
  -F "title=Business Plan" \
  -F "category=business" \
  -F "scope=organization"
```

**Response:**
```json
{
  "id": "abc123",
  "version": 1,
  "status": "ingested",
  "filename": "document.pdf",
  ...
}
```

### 2. Verify Vector Storage

Use the integration test script:

```bash
.\env\Scripts\python.exe backend/src/scripts/test_upload_integration.py abc123 abc123:1
```

**Expected Output:**
```
============================================================
Testing Vector DB Integration
============================================================
Doc ID: abc123
Version: abc123:1
Index: quickstart

--- Testing Query: 'market strategy' ---
✓ Found 3 matches for categories ['go_to_market']
  Score: 0.8234
  Category: go_to_market
  Pages: ['14', '15']

============================================================
✅ SUCCESS: Found 12 total matches
Vector database integration is working!
============================================================
```

### 3. Query from CSO Agent

When the CSO agent needs context:

```python
from src.infrastructure.vector import retrieve_context

# Neo4j determines the truth (what product, what version, what categories)
results = retrieve_context(
    query="What is our go-to-market strategy?",
    doc_id="abc123",
    active_version="abc123:1",
    allowed_categories=["go_to_market", "market", "pricing"],
    top_k=5
)

# Results provide explanatory context for the LLM
for result in results:
    print(f"Context (score {result['score']:.2f}): {result['text']}")
```

## Monitoring & Debugging

### Check Logs

The service logs all vector DB operations:

```
INFO: Starting ingestion for doc_id=abc123, version=abc123:1
INFO: Ingestion complete: 15 segments processed
INFO: Persisting 15 segments to Neo4j (graph DB)
INFO: Neo4j persistence complete
INFO: Persisting 12 segments to Pinecone (vector DB)
INFO: Pinecone persistence complete
```

### Common Issues

#### Issue 1: No embeddings created
**Symptoms:** Upload succeeds but no vector matches found

**Possible Causes:**
1. No text segments in document
2. Pinecone API key not set
3. Pinecone index doesn't exist

**Check Logs:**
```
WARNING: No text segments found for vector storage (doc_id=abc123)
```
or
```
ERROR: Failed to persist to Pinecone: ...
WARNING: Document uploaded successfully but vector embeddings were not created
```

**Solution:**
- Verify `PINECONE_API_KEY` is set in settings
- Verify index exists in Pinecone console
- Check document has extractable text

#### Issue 2: Metadata errors
**Symptoms:** `Bad Request` errors from Pinecone

**Cause:** Invalid metadata format

**Solution:** Already handled! The `writer.py` automatically converts `page_numbers` to strings.

#### Issue 3: Slow uploads
**Symptoms:** Upload takes a long time

**Cause:** Embedding generation for large documents

**Optimization:**
- Embeddings are generated per segment (not per sentence)
- Consider batching for very large documents
- Monitor OpenAI API rate limits

## Testing

### Unit Test: Vector Storage

```bash
.\env\Scripts\python.exe backend/src/scripts/verify_vector.py
```

### Integration Test: Full Upload Flow

1. Start the server:
```bash
cd backend
.\env\Scripts\python.exe -m uvicorn src.main:app --reload
```

2. Upload a test document via API

3. Test vector retrieval:
```bash
.\env\Scripts\python.exe backend/src/scripts/test_upload_integration.py <doc_id> <version_id>
```

## Performance Considerations

### Embedding Generation
- **Model**: `text-embedding-3-large` (3072 dimensions)
- **Cost**: ~$0.00013 per 1K tokens
- **Speed**: ~1-2 seconds per segment

### Typical Document (10 pages)
- **Segments**: ~10-20 segments
- **Embeddings**: ~10-20 embeddings
- **Time**: ~10-30 seconds total
- **Cost**: ~$0.002-0.005

### Optimization Tips
1. **Batch Processing**: Already implemented in `persist_to_pinecone()`
2. **Caching**: Hash-based deduplication (future enhancement)
3. **Async**: Embedding generation is async-ready
4. **Filtering**: Only segments with text are embedded

## Security

### API Keys
- Stored in `settings.py` (loaded from environment)
- Never logged or exposed in responses
- Rotate regularly

### Data Privacy
- Document content is embedded but not stored in Pinecone
- Only metadata and vector representations are stored
- Vectors are not human-readable

### Access Control
- All uploads require authentication
- Organization-level isolation
- Version-based access control

## Future Enhancements

1. **Hash-based Deduplication**
   - Skip re-embedding unchanged segments
   - Reduce costs for document updates

2. **Batch Upload**
   - Process multiple documents in parallel
   - Optimize API calls to OpenAI

3. **Incremental Updates**
   - Only embed changed segments
   - Faster updates for large documents

4. **Metadata Enrichment**
   - Add more context to embeddings
   - Improve retrieval accuracy

5. **Monitoring Dashboard**
   - Track embedding costs
   - Monitor retrieval performance
   - Alert on failures

## Summary

✅ **Vector database is now fully integrated with document uploads**

- Documents are automatically embedded when uploaded
- Both graph (truth) and vector (context) stores are populated
- Error handling ensures uploads never fail due to vector DB issues
- Comprehensive logging for debugging
- Ready for CSO agent integration

The system now maintains the critical separation:
- **Neo4j**: What is true
- **Pinecone**: Why it is true
