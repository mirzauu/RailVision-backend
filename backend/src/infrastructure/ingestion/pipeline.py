from pathlib import Path
from typing import List, Dict
import asyncio
from src.infrastructure.ingestion.loader import load_document
from src.infrastructure.ingestion.segmenter import segment_pages
from src.infrastructure.ingestion.classifier import classify_segment
from src.infrastructure.ingestion.extractor import extract_facts
from src.infrastructure.ingestion.validator import validate_segment


async def run_ingestion(file_path: Path, doc_id: str, version_id: str) -> List[Dict]:
    pages = load_document(file_path)
    segments = segment_pages(pages)
    
    # Limit concurrency to avoid rate limits
    sem = asyncio.Semaphore(5)

    async def process_one(seg: Dict) -> Dict:
        async with sem:
            seg["doc_id"] = doc_id
            seg["doc_version"] = version_id
            if seg.get("page_numbers"):
                seg["segment_id"] = f"{version_id}:page_{seg['page_numbers'][0]}"
            
            # Process sequentially per segment to avoid dict race conditions
            # but segments are processed in parallel
            seg = await classify_segment(seg)
            seg = await extract_facts(seg)
            seg = validate_segment(seg)
            return seg

    processed = await asyncio.gather(*[process_one(s) for s in segments])
    return list(processed)
