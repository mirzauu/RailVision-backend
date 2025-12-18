from pathlib import Path
from typing import List, Dict
import fitz

def load_pdf(path: Path) -> List[Dict]:
    doc = fitz.open(path)
    pages: List[Dict] = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({
                "page_number": i + 1,
                "text": text,
            })
    return pages
