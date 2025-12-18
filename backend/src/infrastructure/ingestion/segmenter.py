from typing import List, Dict

def segment_pages(pages: List[Dict]) -> List[Dict]:
    segments: List[Dict] = []
    for page in pages:
        segments.append({
            "segment_id": f"page_{page['page_number']}",
            "page_numbers": [page["page_number"]],
            "text": page["text"],
        })
    return segments
