import re
import statistics
import fitz  # PyMuPDF

NUM_RE_PDF = re.compile(
    r'^\s*(?P<num>(?:\d+\.)+\d*|\d+|chapter\s+\d+)\b[.\-:)]*\s*(?P<title>.*)',
    re.I
)

def extract_pdf_sections(pdf_path, size_delta_threshold=1.0, dedupe_threshold_ratio=0.6):
    doc = fitz.open(pdf_path)
    all_lines = []

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                line_text = " ".join(
                    span.get("text", "").strip()
                    for span in line.get("spans", [])
                    if span.get("text")
                ).strip()
                if not line_text:
                    continue

                sizes = [span.get("size", 0) for span in line.get("spans", []) if span.get("size")]
                max_size = max(sizes) if sizes else 0

                all_lines.append({
                    "text": line_text,
                    "size": max_size,
                    "page": page_num + 1
                })

    if not all_lines:
        return []

    # Median font size
    sizes = [l["size"] for l in all_lines if l["size"] > 0]
    median_size = statistics.median(sizes) if sizes else 0

    # Candidate headings
    candidates = []
    for l in all_lines:
        starts_num = bool(NUM_RE_PDF.match(l["text"]))
        if l["size"] >= median_size + size_delta_threshold or starts_num:
            candidates.append(l)

    # Deduplicate
    text_counts = {}
    for c in candidates:
        key = c["text"].lower()
        text_counts[key] = text_counts.get(key, 0) + 1
    num_pages = len(doc)
    filtered = [c for c in candidates if text_counts[c["text"].lower()] <= dedupe_threshold_ratio * num_pages]

    # Map font sizes to heading levels
    unique_sizes = sorted({c["size"] for c in filtered}, reverse=True)
    size_to_level = {s: i + 1 for i, s in enumerate(unique_sizes)}

    # Build section list
    sections = []
    seen = set()
    for idx, c in enumerate(filtered):
        key = (c["text"], c["page"])
        if key in seen:
            continue
        seen.add(key)

        # Detect level from numbering
        m = NUM_RE_PDF.match(c["text"])
        num_level = None
        if m:
            num = m.group("num")
            if '.' in num:
                num_level = len(num.split('.'))
            else:
                num_level = 1

        level = size_to_level.get(c["size"], 2)
        if num_level:
            level = min(level, num_level)

        # Collect content until next heading
        content_lines = []
        start_idx = all_lines.index(c)
        end_idx = len(all_lines)
        if idx + 1 < len(filtered):
            end_idx = all_lines.index(filtered[idx + 1])

        for l in all_lines[start_idx + 1:end_idx]:
            content_lines.append(l["text"])

        sections.append({
            "title": c["text"],      # keep numbering
            "level": int(level),
            "page": c["page"],
            "content": " ".join(content_lines).strip()
        })

    return sections
