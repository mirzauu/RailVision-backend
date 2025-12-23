import re

# Common corporate suffixes to strip
_SUFFIXES = [
    r"\s+inc\.?$", r"\s+ltd\.?$", r"\s+llc\.?$", r"\s+corp\.?$", 
    r"\s+corporation$", r"\s+company$", r"\s+gmbh$", r"\s+plc\.?$", 
    r"\s+sa$", r"\s+s\.a\.$", r"\s+limited$"
]

def normalize_name(name: str) -> str:
    """
    Normalizes an entity name for resolution.
    1. Lowercase
    2. Strip whitespace
    3. Remove common corporate suffixes
    4. Remove special characters (keep alphanumeric and spaces)
    """
    if not name:
        return ""
    
    # Lowercase and strip
    norm = name.lower().strip()
    
    # Strip suffixes
    for suffix in _SUFFIXES:
        norm = re.sub(suffix, "", norm)
    
    # Remove special chars (keep spaces)
    norm = re.sub(r"[^a-z0-9\s]", "", norm)
    
    # Collapse multiple spaces
    norm = re.sub(r"\s+", " ", norm).strip()
    
    return norm
