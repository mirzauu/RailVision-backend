from typing import List, Dict

def segment_pages(pages: List[Dict], chunk_size: int = 4000, overlap: int = 400) -> List[Dict]:
    """
    Segment pages into smaller chunks to fit within embedding model context limits.
    Roughly 4 chars ~= 1 token. 4000 chars is ~1000 tokens.
    """
    segments: List[Dict] = []
    
    for page in pages:
        text = page.get("text", "")
        if not text:
            continue
            
        page_num = page.get("page_number", 0)
        
        # If text is small enough, keep as one segment
        if len(text) <= chunk_size:
            segments.append({
                "segment_id": f"page_{page_num}_part_1",
                "page_numbers": [page_num],
                "text": text,
            })
            continue
            
        # Split text
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            
            # If we are at the end, just take the rest
            if end >= text_len:
                chunks.append(text[start:])
                break
            
            # intelligent split: try to find a sentence or paragraph break
            # look for \n\n, then \n, then . (space), then space
            # we look backwards from 'end'
            best_boundary = -1
            search_window_start = max(start + chunk_size // 2, end - 500)
            
            # Prioritize paragraph breaks
            boundary = text.rfind('\n\n', search_window_start, end)
            if boundary != -1:
                best_boundary = boundary + 2 # include the newlines
            
            if best_boundary == -1:
                boundary = text.rfind('\n', search_window_start, end)
                if boundary != -1:
                    best_boundary = boundary + 1
            
            if best_boundary == -1:
                 # look for sentence end
                boundary = text.rfind('. ', search_window_start, end)
                if boundary != -1:
                    best_boundary = boundary + 2
            
            if best_boundary == -1:
                boundary = text.rfind(' ', search_window_start, end)
                if boundary != -1:
                    best_boundary = boundary + 1
            
            # If no good boundary (e.g. huge string with no spaces), hard chop
            if best_boundary == -1:
                best_boundary = end
                
            chunks.append(text[start:best_boundary])
            
            # Move start for next chunk
            # If we chopped at 'best_boundary', the next chunk starts there minus overlap (unless we are just hard chopping)
            # Actually standard practice is:
            # Chunk 1: [0 : best_boundary]
            # Chunk 2: [best_boundary - overlap : ...]
            # But simpler is:
            # start = best_boundary (no overlap explicitly carried over from previous, 
            # BUT we want overlap.
            # So let's just slide the window.
            
            # Alternative Loop:
            # just take text[start:end] where end is strictly start + chunk_size
            # then find boundary within that window to verify
            # then set start = boundary - overlap? No that's complicated.
            
            # Let's sticks to: next chunk starts at `best_boundary`.
            # To simulate overlap, we should have backed up `start`?
            # actually, 'overlap' means the previous chunk shares some text with this one.
            # So if we cut at `best_boundary`, the next chunk should start at `best_boundary - overlap`.
            # But `best_boundary` is where we CUT.
            
            start = best_boundary - overlap
            if start < 0: start = 0 # Should not happen unless chunk is tiny
            
            # Avoid getting stuck: if start didn't advance enough (less than 10 chars), force advance
            # previous start was 'old_start'
            # we need to ensure we are moving forward.
            # Wait, this loop logic is getting complex.
            
        # Recursive Character splitter logic is better but let's stick to a simpler sliding window 
        # that handles overlaps correctly.
        
        chunks = []
        cursors = 0
        while cursors < text_len:
            # Take a chunk of size `chunk_size`
            chunk_end = min(cursors + chunk_size, text_len)
            chunk_text = text[cursors:chunk_end]
            
            # If we are not at the end of text, we want to find a good break point within the last 10% of the chunk
            # or just use the end
            if chunk_end < text_len:
                # Find break in the last portion
                limit = max(0, len(chunk_text) - overlap) 
                # actually we want to cut somewhat before the hard limit so the next chunk has context
                # But typically we cut at a sentence, and then REWIND for the next chunk?
                # No, standard overlapping:
                # Chunk 1: 0 to 1000
                # Chunk 2: 800 to 1800 (200 overlap)
                
                # So we just cut at 'chunk_end' (hard limit) or a 'nice break' nearby?
                # If we rely on overlap, we can just hard cut, and the overlap provides continuity.
                # But it's better to cut at clean boundaries.
                
                pass 
                
            # LET'S USE A SIMPLE APPROACH:
            # Just slide by (chunk_size - overlap)
            # And clean up the boundaries if possible?
            
            chunks.append(chunk_text)
            
            if chunk_end == text_len:
                break
                
            cursors += (chunk_size - overlap)
            
        # Optimize chunks (cleaning boundaries)
        # That logic is a bit purely strictly positional.
        
        for i, chunk_txt in enumerate(chunks):
            segments.append({
                "segment_id": f"page_{page_num}_part_{i+1}",
                "page_numbers": [page_num],
                "text": chunk_txt,
            })
            
    return segments
