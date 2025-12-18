from pathlib import Path
from typing import List, Dict
from src.infrastructure.ingestion.loader import load_pdf
from src.infrastructure.ingestion.segmenter import segment_pages
from src.infrastructure.ingestion.classifier import classify_segment
from src.infrastructure.ingestion.extractor import extract_facts
from src.infrastructure.ingestion.validator import validate_segment

async def run_ingestion(file_path: Path, doc_id: str, version_id: str) -> List[Dict]:
    pages = load_pdf(file_path)
    segments = segment_pages(pages)
    processed: List[Dict] = []
    for seg in segments:
        seg["doc_id"] = doc_id
        seg["doc_version"] = version_id
        if seg.get("page_numbers"):
            seg["segment_id"] = f"{version_id}:page_{seg['page_numbers'][0]}"
        seg = await classify_segment(seg)
        seg = await extract_facts(seg)
        seg = validate_segment(seg)
        processed.append(seg)
    return processed
