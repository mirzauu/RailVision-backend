import fitz
from docx import Document
from pathlib import Path
from typing import List, Dict

def load_pdf(path: Path) -> List[Dict]:
    doc = fitz.open(str(path))

    pages: List[Dict] = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({
                "page_number": i + 1,
                "text": text,
            })
    return pages

def load_docx(path: Path) -> List[Dict]:
    import zipfile
    if not zipfile.is_zipfile(path):
        # Fallback to text if it's named .docx but isn't a zip
        return load_text(path)
        
    try:
        doc = Document(str(path))
        text = "\n".join([para.text for para in doc.paragraphs])
        if text.strip():
            return [{"page_number": 1, "text": text.strip()}]
    except Exception as e:
        print(f"Error loading docx {path}: {e}")
        return load_text(path)
    return []

def load_text(path: Path) -> List[Dict]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1").strip()
    
    if text:
        return [{"page_number": 1, "text": text}]
    return []

def load_document(path: Path) -> List[Dict]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    elif ext in (".txt", ".md"):
        return load_text(path)
    # Default to text if unknown
    return load_text(path)
