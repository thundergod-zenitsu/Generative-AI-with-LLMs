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


#=======================================================================================================================================
import re
import statistics
import fitz  # PyMuPDF

NUM_RE_PDF = re.compile(
    r'^\s*(?P<num>(?:\d+\.)+\d*|\d+|chapter\s+\d+)\b[.\-:)]*\s*(?P<title>.*)',
    re.I
)

def extract_pdf_sections(pdf_path, size_delta_threshold=1.0, dedupe_threshold_ratio=0.6, footnote_size_ratio=0.7):
    """
    Extract structured sections (title + content) from a PDF,
    while ignoring footnotes based on font size and position.
    """
    doc = fitz.open(pdf_path)
    all_lines = []

    for page_num, page in enumerate(doc):
        page_height = page.rect.height
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

                # Take y-position of line (first span's bbox)
                y_pos = line["spans"][0]["bbox"][1] if line["spans"] else 0

                all_lines.append({
                    "text": line_text,
                    "size": max_size,
                    "page": page_num + 1,
                    "y": y_pos,
                    "page_height": page_height
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

    # Deduplicate repeated headers/footers
    text_counts = {}
    for c in candidates:
        key = c["text"].lower()
        text_counts[key] = text_counts.get(key, 0) + 1
    num_pages = len(doc)
    filtered = [c for c in candidates if text_counts[c["text"].lower()] <= dedupe_threshold_ratio * num_pages]

    # Map font sizes to heading levels
    unique_sizes = sorted({c["size"] for c in filtered}, reverse=True)
    size_to_level = {s: i + 1 for i, s in enumerate(unique_sizes)}

    sections = []
    seen = set()
    for idx, c in enumerate(filtered):
        key = (c["text"], c["page"])
        if key in seen:
            continue
        seen.add(key)

        # Determine heading level
        m = NUM_RE_PDF.match(c["text"])
        num_level = None
        if m:
            num = m.group("num")
            num_level = len(num.split('.')) if '.' in num else 1

        level = size_to_level.get(c["size"], 2)
        if num_level:
            level = min(level, num_level)

        # Collect section content until next heading
        content_lines = []
        start_idx = all_lines.index(c)
        end_idx = len(all_lines)
        if idx + 1 < len(filtered):
            end_idx = all_lines.index(filtered[idx + 1])

        for l in all_lines[start_idx + 1:end_idx]:
            # --- Footnote filter ---
            too_small = l["size"] < footnote_size_ratio * median_size
            near_bottom = l["y"] > 0.9 * l["page_height"]
            looks_like_note = re.match(r'^\s*[\d\*\†]+\s', l["text"])
            if too_small or near_bottom or looks_like_note:
                continue
            # -----------------------
            content_lines.append(l["text"])

        sections.append({
            "title": c["text"],
            "level": int(level),
            "page": c["page"],
            "content": " ".join(content_lines).strip()
        })

    return sections
#=======================================================================================================================================

import re
import statistics
import fitz  # PyMuPDF

NUM_RE_PDF = re.compile(
    r'^\s*(?P<num>(?:\d+\.)+\d*|\d+|chapter\s+\d+)\b[.\-:)]*\s*(?P<title>.*)',
    re.I
)

def extract_pdf_sections_no_footnotes(pdf_path, size_delta_threshold=1.0, dedupe_threshold_ratio=0.6):
    doc = fitz.open(pdf_path)
    all_lines = []

    for page_num, page in enumerate(doc):
        page_height = page.rect.height
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

                # average y position of line
                y_positions = [span.get("bbox")[1] for span in line.get("spans", []) if "bbox" in span]
                avg_y = sum(y_positions) / len(y_positions) if y_positions else 0

                all_lines.append({
                    "text": line_text,
                    "size": max_size,
                    "page": page_num + 1,
                    "y": avg_y,
                    "page_height": page_height
                })

    if not all_lines:
        return []

    # Median size
    sizes = [l["size"] for l in all_lines if l["size"] > 0]
    median_size = statistics.median(sizes) if sizes else 0

    # Filter out likely footnotes (small font or bottom of page)
    body_lines = []
    for l in all_lines:
        if l["size"] < median_size - 1.5:  # much smaller than normal text
            continue
        if l["y"] > 0.9 * l["page_height"]:  # bottom 10% of page
            continue
        body_lines.append(l)

    # Candidate headings
    candidates = []
    for l in body_lines:
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
        start_idx = body_lines.index(c)
        end_idx = len(body_lines)
        if idx + 1 < len(filtered):
            end_idx = body_lines.index(filtered[idx + 1])

        for l in body_lines[start_idx + 1:end_idx]:
            content_lines.append(l["text"])

        sections.append({
            "title": c["text"],      # keep numbering
            "level": int(level),
            "page": c["page"],
            "content": " ".join(content_lines).strip()
        })

    return sections

#=======================================================================================================================================
import re
import statistics
import fitz  # PyMuPDF

NUM_RE_PDF = re.compile(
    r'^\s*(?P<num>(?:\d+\.)+\d*|\d+)\b[.\-:)]*\s*(?P<title>.*)',
    re.I
)

def parse_numbering(num_str):
    """Convert '2.3.1' -> [2, 3, 1] as integers"""
    return [int(x) for x in num_str.strip('.').split('.') if x.isdigit()]

def is_next_section(prev_nums, curr_nums):
    """
    Decide if curr_nums is a valid next section after prev_nums.
    Examples:
    prev [2] -> curr [3] ✅ new main section
    prev [2,1] -> curr [2,2] ✅ next subsection
    prev [2,1] -> curr [2,1,1] ✅ nested subsection
    prev [2,2] -> curr [3] ✅ new chapter
    """
    if not prev_nums:
        return True
    # Same depth → should increment last number
    if len(prev_nums) == len(curr_nums):
        return curr_nums[:-1] == prev_nums[:-1] and curr_nums[-1] == prev_nums[-1] + 1
    # One deeper → child section
    if len(curr_nums) == len(prev_nums) + 1:
        return curr_nums[:-1] == prev_nums and curr_nums[-1] == 1
    # One higher → closing subsection, allow reset
    if len(curr_nums) < len(prev_nums):
        return curr_nums[-1] == prev_nums[len(curr_nums)-1] + 1 or curr_nums[-1] == 1
    return False

def extract_pdf_sections_no_footnotes(pdf_path, size_delta_threshold=1.0, dedupe_threshold_ratio=0.6):
    doc = fitz.open(pdf_path)
    all_lines = []

    for page_num, page in enumerate(doc):
        page_height = page.rect.height
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

                # average y position of line
                y_positions = [span.get("bbox")[1] for span in line.get("spans", []) if "bbox" in span]
                avg_y = sum(y_positions) / len(y_positions) if y_positions else 0

                all_lines.append({
                    "text": line_text,
                    "size": max_size,
                    "page": page_num + 1,
                    "y": avg_y,
                    "page_height": page_height
                })

    if not all_lines:
        return []

    # Median size
    sizes = [l["size"] for l in all_lines if l["size"] > 0]
    median_size = statistics.median(sizes) if sizes else 0

    # Filter out likely footnotes
    body_lines = []
    for l in all_lines:
        if l["size"] < median_size - 1.5:  # much smaller font
            continue
        if l["y"] > 0.9 * l["page_height"]:  # bottom 10% of page
            continue
        body_lines.append(l)

    sections = []
    current_section = None
    prev_nums = None

    for i, l in enumerate(body_lines):
        m = NUM_RE_PDF.match(l["text"])
        is_heading = False
        curr_nums = None

        if m:
            curr_nums = parse_numbering(m.group("num"))
            # Validate section numbering sequence
            if prev_nums is None or is_next_section(prev_nums, curr_nums):
                is_heading = True
        elif l["size"] >= median_size + size_delta_threshold:
            # Large font non-numbered heading
            is_heading = True

        if is_heading:
            # Close previous section
            if current_section:
                sections.append(current_section)

            current_section = {
                "title": l["text"],
                "page": l["page"],
                "content": "",
                "numbering": curr_nums or []
            }
            prev_nums = curr_nums
        else:
            if current_section:
                current_section["content"] += " " + l["text"]

    # Append last section
    if current_section:
        sections.append(current_section)

    return sections

#=======================================================================================================================================