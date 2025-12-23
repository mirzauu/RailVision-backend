from typing import List, Dict, Any
import logging
from src.infrastructure.vector.client import get_index
from src.infrastructure.vector.embedder import embed_texts, embed_text

logger = logging.getLogger(__name__)

def upsert_segments_batch(segments: List[Dict[str, Any]], batch_size: int = 100):
    if not segments:
        logger.warning("No segments provided for Pinecone upsert")
        return
        
    try:
        index = get_index()
    except Exception as e:
        logger.error(f"Failed to get Pinecone index: {e}")
        raise
    
    logger.info(f"Starting Pinecone upsert for {len(segments)} segments")

    # 1. Prepare data
    ids = []
    texts = []
    metadatas = []
    
    for seg in segments:
        try:
            # Robust ID construction
            seg_id = f"{seg['doc_id']}:{seg['doc_version']}:{seg.get('segment_id', '')}"
            
            # Metadata handling
            meta = {
                "doc_id": seg["doc_id"],
                "doc_version": seg["doc_version"],
                "category": seg["category"],
                "page_numbers": seg["page_numbers"],
                "confidence": seg.get("classification_confidence", 0.8),
                # Add text to metadata for retrieval context
                "text": seg["text"] 
            }
            
            # Pinecone requirement: lists must be strings
            if "page_numbers" in meta and isinstance(meta["page_numbers"], list):
                meta["page_numbers"] = [str(p) for p in meta["page_numbers"]]
                
            ids.append(seg_id)
            texts.append(seg["text"])
            metadatas.append(meta)
        except KeyError as e:
            logger.error(f"Missing key in segment during preparation: {e}")
            continue

    if not ids:
        logger.warning("No valid segments prepared for upsert")
        return

    # 2. Embed in batches (OpenAI limit)
    # 20 is a safe batch size for large text chunks
    vectors = []
    embed_batch_size = 20 
    logger.info(f"Generating embeddings for {len(texts)} texts in batches of {embed_batch_size}")
    
    try:
        for i in range(0, len(texts), embed_batch_size):
            batch_texts = texts[i : i + embed_batch_size]
            batch_vectors = embed_texts(batch_texts)
            vectors.extend(batch_vectors)
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

    if len(vectors) != len(ids):
        logger.error(f"Mismatch between ID count ({len(ids)}) and vector count ({len(vectors)})")
        return

    # 3. Upsert to Pinecone in batches
    # Pinecone recommends batches of 100 or less
    total = len(ids)
    upserted_count = 0
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch_vectors_payload = []
        for j in range(i, end):
            batch_vectors_payload.append({
                "id": ids[j],
                "values": vectors[j],
                "metadata": metadatas[j]
            })
        
        if batch_vectors_payload:
            try:
                index.upsert(vectors=batch_vectors_payload)
                upserted_count += len(batch_vectors_payload)
                logger.info(f"Upserted batch {i//batch_size + 1}: {len(batch_vectors_payload)} vectors")
            except Exception as e:
                logger.error(f"Failed to upsert batch starting at index {i}: {e}")
                # Don't raise, try next batch
                continue
    
    logger.info(f"Pinecone upsert complete. Total vectors upserted: {upserted_count}/{total}")

def persist_to_pinecone(processed_segments: list[dict]):
    upsert_segments_batch(processed_segments)

def upsert_segment(segment_id: str, text: str, metadata: dict):
    # Legacy support / Single item upsert
    index = get_index()
    vector = embed_text(text)
    
    if "page_numbers" in metadata and isinstance(metadata["page_numbers"], list):
        metadata["page_numbers"] = [str(p) for p in metadata["page_numbers"]]
        
    index.upsert([{
        "id": segment_id,
        "values": vector,
        "metadata": metadata
    }])
