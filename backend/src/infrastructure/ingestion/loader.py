import fitz
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
    import xml.etree.ElementTree as ET
    if not zipfile.is_zipfile(path):
        return load_text(path)
    try:
        with zipfile.ZipFile(path) as z:
            with z.open("word/document.xml") as f:
                data = f.read()
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        root = ET.fromstring(data)
        body = root.find(f".//{ns}body")
        if body is None:
            return load_text(path)
        pages: List[Dict] = []
        buf: List[str] = []
        page_num = 1
        for p in list(body):
            if p.tag != f"{ns}p":
                continue
            for child in list(p):
                if child.tag == f"{ns}r":
                    for sub in list(child):
                        if sub.tag == f"{ns}t":
                            buf.append(sub.text or "")
                        elif sub.tag == f"{ns}br":
                            br_type = sub.attrib.get(f"{ns}type")
                            if br_type == "page":
                                text = "".join(buf).strip()
                                if text:
                                    pages.append({"page_number": page_num, "text": text})
                                page_num += 1
                                buf = []
            buf.append("\n")
        remaining = "".join(buf).strip()
        if remaining:
            pages.append({"page_number": page_num, "text": remaining})
        if pages:
            return pages
        return load_text(path)
    except Exception as e:
        print(f"Error loading docx {path}: {e}")
        return load_text(path)

def load_text(path: Path) -> List[Dict]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1").strip()
    
    if text:
        return [{"page_number": 1, "text": text}]
    return []

def load_pptx(path: Path) -> List[Dict]:
    import zipfile
    import re
    import xml.etree.ElementTree as ET
    if not zipfile.is_zipfile(path):
        return load_text(path)
    try:
        slides = []
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            def slide_index(name: str) -> int:
                m = re.search(r"slide(\d+)\.xml$", name)
                return int(m.group(1)) if m else 0
            names.sort(key=slide_index)
            for i, name in enumerate(names, start=1):
                with z.open(name) as f:
                    data = f.read()
                root = ET.fromstring(data)
                texts = []
                for elem in root.iter():
                    tag = elem.tag
                    if tag.endswith("}t"):
                        if elem.text:
                            texts.append(elem.text)
                content = "\n".join(texts).strip()
                if content:
                    slides.append({"page_number": i, "text": content})
        if slides:
            return slides
        return load_text(path)
    except Exception as e:
        print(f"Error loading pptx {path}: {e}")
        return load_text(path)

def load_xlsx(path: Path) -> List[Dict]:
    try:
        import pandas as pd
        dfs = pd.read_excel(path, sheet_name=None) # Read all sheets
        pages = []
        for i, (sheet_name, df) in enumerate(dfs.items(), 1):
            text = f"Sheet: {sheet_name}\n" + df.to_string(index=False)
            pages.append({"page_number": i, "text": text})
        return pages
    except Exception as e:
        print(f"Error loading xlsx {path}: {e}")
        return load_text(path)

def load_csv(path: Path) -> List[Dict]:
    try:
        import pandas as pd
        df = pd.read_csv(path)
        text = df.to_string(index=False)
        return [{"page_number": 1, "text": text}]
    except Exception as e:
        print(f"Error loading csv {path}: {e}")
        return load_text(path)

def load_document(path: Path) -> List[Dict]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    elif ext == ".pptx":
        return load_pptx(path)
    elif ext == ".xlsx":
        return load_xlsx(path)
    elif ext == ".csv":
        return load_csv(path)
    elif ext in (".txt", ".md"):
        return load_text(path)
    # Default to text if unknown
    return load_text(path)
