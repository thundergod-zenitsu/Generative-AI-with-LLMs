"""
page_index_tree_full.py

Features:
- PDF layout extraction (pdfplumber) -> pages with bounding boxes for text blocks.
- Token-aware chunking using tiktoken.
- Parallel LLM summarization and title detection using OpenAI ChatCompletion (configurable model).
- Rate-limit & retry using tenacity and concurrent.Semaphore.
- Intermediate parent node persistence to disk (JSON per node) for restartability.
- Nested export and flat-level export (MCP-like top-level).
"""

import os
import json
import time
import uuid
import math
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

import pdfplumber
import tiktoken
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

# --- Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PageIndexTree")

# --- Constants / Defaults
DEFAULT_MODEL = "gpt-4o-mini"     # change if needed
DEFAULT_MAX_TOKENS_SUMMARY = 1024 # max tokens you expect the summary to consume (for prompt budgeting)
DEFAULT_CHUNK_TOKENS = 700       # tokens per leaf chunk (adjustable)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment. Set OPENAI_API_KEY before running.")
openai.api_key = OPENAI_API_KEY

# --- helpers
def deterministic_id(prefix: str, counter: int, width: int = 4) -> str:
    return f"{prefix}{str(counter).zfill(width)}"

def hash_text(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()

# --- Node dataclass
@dataclass
class TreeNode:
    id: str
    content: str                        # summary (for internal nodes) or original chunk text (for leaf)
    level: int                          # 0 = leaf
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    title: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # includes bounding boxes for leaves

    def to_dict(self, nodes_map: Dict[str, "TreeNode"], nested: bool = True) -> Dict[str, Any]:
        base = {
            "id": self.id,
            "title": self.title,
            "summary": self.content,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "level": self.level,
            "metadata": self.metadata
        }
        if nested:
            base["children"] = [nodes_map[c].to_dict(nodes_map, nested=True) for c in self.children]
        else:
            base["children"] = self.children[:]
        return base

# --- OpenAI call wrappers with rate-limit/retry
class OpenAIClient:
    def __init__(self, model: str = DEFAULT_MODEL, max_concurrent_calls: int = 4, max_retries: int = 6):
        self.model = model
        self.semaphore = Semaphore(max_concurrent_calls)  # global concurrency limit
        self.max_retries = max_retries

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception)
    )
    def _call_chat(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> str:
        """
        Basic wrapper around openai.ChatCompletion.create with exponential backoff.
        The retry decorator will retry on exceptions (network, 429, 5xx).
        We also gate concurrent calls with a semaphore.
        """
        acquired = self.semaphore.acquire(timeout=120)
        if not acquired:
            raise TimeoutError("Could not acquire semaphore for OpenAI calls (timeout).")
        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            # Extract content
            choice = resp.choices[0]
            content = choice.message["content"] if "message" in choice else choice["text"]
            return content.strip()
        finally:
            # small sleep to avoid burst releasing causing spikes
            time.sleep(0.1)
            self.semaphore.release()

    def summarize(self, texts: List[str], prompt_template: Optional[str] = None, max_tokens: int = DEFAULT_MAX_TOKENS_SUMMARY) -> str:
        """
        Summarize a list of texts into a single short summary.
        """
        joined = "\n\n".join(texts)
        if not prompt_template:
            prompt_template = (
                "You are a concise summarization assistant. Summarize the following passages into a single "
                "short, accurate, and informative section summary (3-6 sentences). Focus on what the section is about.\n\n"
                "Passages:\n{passages}\n\nSummary:"
            )
        prompt = prompt_template.format(passages=joined)
        messages = [{"role":"user", "content": prompt}]
        try:
            return self._call_chat(messages, max_tokens=max_tokens, temperature=0.0)
        except RetryError as e:
            logger.exception("OpenAI summarize RetryError")
            # fallback: return a truncated concatenation
            return joined[:2000]

    def detect_title(self, texts: List[str], prompt_template: Optional[str] = None, max_tokens: int = 64) -> Optional[str]:
        """
        Detect concise title/heading for a group of passages.
        Returns short phrase or None.
        """
        joined = "\n\n".join(texts)
        if not prompt_template:
            prompt_template = (
                "Given the following passages that belong to the same document section, return the most likely "
                "section heading/title in 1 short phrase (no punctuation). If none, return an empty string.\n\n"
                "Passages:\n{passages}\n\nTitle:"
            )
        prompt = prompt_template.format(passages=joined)
        messages = [{"role":"user", "content": prompt}]
        try:
            out = self._call_chat(messages, max_tokens=max_tokens, temperature=0.0)
            out = out.strip()
            if not out:
                return None
            # clean common artifacts
            out = out.strip('" \n\r\t')
            # If LLM returns more than a short phrase, truncate to first line/phrase
            out = out.splitlines()[0]
            if len(out.split()) > 12:
                out = " ".join(out.split()[:12])
            return out
        except RetryError:
            return None

# --- Token-based chunker using tiktoken
class TokenChunker:
    def __init__(self, model_name: str = DEFAULT_MODEL, chunk_token_size: int = DEFAULT_CHUNK_TOKENS):
        # choose encoding that matches model family; fall back to cl100k_base
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chunk_token_size = chunk_token_size

    def tokens(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def token_len(self, text: str) -> int:
        return len(self.tokens(text))

    def chunk_text(self, text: str, max_tokens: Optional[int] = None) -> List[str]:
        """
        Split text into chunks <= max_tokens (or self.chunk_token_size) using the tokenizer.
        Returns list of string chunks.
        Strategy: attempt to split on paragraph breaks, then fall back to token slices.
        """
        if max_tokens is None:
            max_tokens = self.chunk_token_size

        # split on double newlines first (paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""
        current_tokens = 0

        for p in paragraphs:
            p_tokens = self.token_len(p)
            if current_tokens + p_tokens <= max_tokens:
                current = (current + "\n\n" + p).strip() if current else p
                current_tokens += p_tokens
            else:
                # if p itself > max_tokens, hard-split it via tokens
                if p_tokens > max_tokens:
                    # flush current
                    if current:
                        chunks.append(current)
                        current = ""
                        current_tokens = 0
                    # break p into token slices
                    tokens = self.tokens(p)
                    for i in range(0, len(tokens), max_tokens):
                        chunk_tokens = tokens[i:i+max_tokens]
                        chunk_text = self.encoding.decode(chunk_tokens)
                        chunks.append(chunk_text)
                else:
                    # flush current and start new
                    if current:
                        chunks.append(current)
                    current = p
                    current_tokens = p_tokens
        if current:
            chunks.append(current)
        # final guard: ensure no chunk exceeds max_tokens
        final_chunks = []
        for c in chunks:
            if self.token_len(c) <= max_tokens:
                final_chunks.append(c)
            else:
                # token-slice
                tokens = self.tokens(c)
                for i in range(0, len(tokens), max_tokens):
                    final_chunks.append(self.encoding.decode(tokens[i:i+max_tokens]))
        return final_chunks

# --- PDF layout extraction (pdfplumber) -> pages list with blocks and bboxes
def extract_pages_with_layout(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Returns list (page_indexed from 1) of dicts:
    {
      "page_number": int,
      "blocks": [ { "text": "...", "bbox": (x0,y0,x1,y1), "x0":..., "x1":..., "top":..., "bottom":..., "fontname":..., "size":... } , ... ],
      "raw_text": "...",  # entire page text
    }
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, pg in enumerate(pdf.pages, start=1):
            blocks = []
            # pdfplumber.extract_words gives word-level with bbox and font size maybe absent; we'll group into lines/blocks using .extract_text with layout
            try:
                words = pg.extract_words(use_text_flow=True)
                # group words into simple lines by 'top' coordinate
                lines = {}
                for w in words:
                    top_key = round(float(w.get("top", 0)), 1)
                    lines.setdefault(top_key, []).append(w)
                for top_key in sorted(lines.keys()):
                    ws = lines[top_key]
                    text = " ".join(w["text"] for w in ws)
                    # approximate bounding box as min/max of words
                    x0 = min(float(w["x0"]) for w in ws)
                    x1 = max(float(w["x1"]) for w in ws)
                    top = min(float(w.get("top", 0)) for w in ws)
                    bottom = max(float(w.get("bottom", 0)) for w in ws)
                    blocks.append({
                        "text": text,
                        "bbox": (x0, top, x1, bottom),
                        "x0": x0, "x1": x1, "top": top, "bottom": bottom,
                        "fontname": ws[0].get("fontname"),
                        "size": ws[0].get("size")
                    })
            except Exception:
                # fallback to whole page text as one block
                text = pg.extract_text() or ""
                blocks.append({
                    "text": text,
                    "bbox": (0,0,0,0),
                    "x0":0,"x1":0,"top":0,"bottom":0,"fontname":None,"size":None
                })
            pages.append({"page_number": i, "blocks": blocks, "raw_text": pg.extract_text() or ""})
    return pages

# --- PageIndexTree implementation
class PageIndexTree:
    def __init__(
        self,
        openai_client: OpenAIClient,
        chunker: TokenChunker,
        group_size: int = 6,
        max_workers: int = 6,
        nodes_dir: str = "pageindex_nodes",
        id_prefix: str = "N"
    ):
        self.openai = openai_client
        self.chunker = chunker
        self.group_size = group_size
        self.max_workers = max_workers
        self.nodes: Dict[str, TreeNode] = {}
        self.root_id: Optional[str] = None
        self._counter = 0
        self.id_prefix = id_prefix
        self.nodes_dir = nodes_dir
        os.makedirs(self.nodes_dir, exist_ok=True)
        # caches to avoid duplicate LLM calls across runs (in-memory)
        self._summary_cache = {}  # key -> summary
        self._title_cache = {}    # key -> title

    def _persist_node(self, node: TreeNode):
        path = os.path.join(self.nodes_dir, f"node_{node.id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(node), f, ensure_ascii=False, indent=2)

    def _load_persisted_nodes(self):
        """Load any nodes already saved in nodes_dir into self.nodes (useful for resume)."""
        for fname in os.listdir(self.nodes_dir):
            if not fname.startswith("node_") or not fname.endswith(".json"):
                continue
            path = os.path.join(self.nodes_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    node = TreeNode(**obj)
                    self.nodes[node.id] = node
                    # increment internal counter from numeric suffix if possible
                    try:
                        num = int(node.id.lstrip(self.id_prefix))
                        if num > self._counter:
                            self._counter = num
                    except Exception:
                        pass
            except Exception:
                logger.exception("Failed to load persisted node %s", path)

    def _make_node(self, content: str, level: int, start_page: int = None, end_page: int = None, children: List[str] = None, metadata: Dict = None) -> str:
        self._counter += 1
        node_id = deterministic_id(self.id_prefix, self._counter, width=6)
        node = TreeNode(
            id=node_id,
            content=content,
            level=level,
            children=list(children) if children else [],
            parent=None,
            title=None,
            start_page=start_page,
            end_page=end_page,
            metadata=metadata or {}
        )
        self.nodes[node_id] = node
        # persist to disk immediately for restartability
        try:
            self._persist_node(node)
        except Exception:
            logger.exception("Failed persisting node %s", node_id)
        return node_id

    # --- convert pages-with-layout into leaf nodes (token-aware, with bounding boxes)
    def pages_to_leaves(self, pages_with_layout: List[Dict[str, Any]], max_chunk_tokens: Optional[int] = None) -> List[str]:
        """
        pages_with_layout: output from extract_pages_with_layout
        For each page, group blocks into text, then chunk via token-chunker.
        Each leaf stores bounding boxes (list) and page number in metadata.
        """
        if max_chunk_tokens is None:
            max_chunk_tokens = self.chunker.chunk_token_size

        leaf_ids = []
        for page in pages_with_layout:
            pg_num = page["page_number"]
            # Build a sequence of block texts with bbox metadata
            blocks = page.get("blocks", [])
            # Simple heuristic: group blocks into a single page_text but keep block bboxes for mapping
            page_text = ""
            block_map = []
            for b in blocks:
                text = (b.get("text") or "").strip()
                if not text:
                    continue
                # keep mapping: length of added text -> bbox
                start_char = len(page_text)
                page_text += (("\n\n" + text) if page_text else text)
                end_char = len(page_text)
                block_map.append({"start": start_char, "end": end_char, "bbox": b.get("bbox"), "meta": {"x0":b.get("x0"), "x1":b.get("x1"), "top":b.get("top"), "bottom":b.get("bottom"), "size":b.get("size")}})

            if not page_text.strip():
                # create empty leaf to preserve page range
                nid = self._make_node(content="", level=0, start_page=pg_num, end_page=pg_num, metadata={"page": pg_num, "bboxes": []})
                leaf_ids.append(nid)
                continue

            # Token-based chunking on page_text
            chunks = self.chunker.chunk_text(page_text, max_tokens=max_chunk_tokens)

            # For each chunk, map subranges of chars back to bboxes by approximate char positions
            char_cursor = 0
            for chunk in chunks:
                # find where chunk occurs in page_text — we use char cursor to locate
                loc = page_text.find(chunk, char_cursor)
                if loc == -1:
                    # fallback: advance cursor and find anywhere
                    loc = page_text.find(chunk)
                if loc == -1:
                    # cannot map — keep whole page bboxes
                    bboxes = [bm["bbox"] for bm in block_map]
                else:
                    chunk_start = loc
                    chunk_end = loc + len(chunk)
                    # collect bboxes overlapping this span
                    bboxes = []
                    for bm in block_map:
                        if not (bm["end"] < chunk_start or bm["start"] > chunk_end):
                            bboxes.append(bm["bbox"])
                    if not bboxes:
                        bboxes = [bm["bbox"] for bm in block_map] or []
                    char_cursor = chunk_end

                metadata = {"page": pg_num, "bboxes": bboxes}
                nid = self._make_node(content=chunk, level=0, start_page=pg_num, end_page=pg_num, metadata=metadata)
                leaf_ids.append(nid)

        return leaf_ids

    # --- Build hierarchical tree bottom-up with parallel LLM summarization and title detection; persist parent nodes
    def build_tree_parallel(self, leaf_ids: List[str], prefer_title_on_parent: bool = True, resume: bool = True, progress: bool = False) -> str:
        """
        leaf_ids: list of leaf node IDs (level 0)
        resume: if True, will load persisted nodes and skip building nodes that already exist (safe restarts)
        """
        if resume:
            self._load_persisted_nodes()

        current_level_ids = [nid for nid in leaf_ids if self.nodes.get(nid)]  # ensure exist
        # if persisted nodes exist that are at higher levels, drop leaves that are already attached (we keep simple)
        # sort leaves by page and original creation order (deterministic)
        # we'll ascend until one root remains
        current_level = 0
        while len(current_level_ids) > 1:
            logger.info("Building level %d with %d nodes", current_level+1, len(current_level_ids))
            # create groups in order
            groups = [current_level_ids[i:i+self.group_size] for i in range(0, len(current_level_ids), self.group_size)]
            summaries = {}
            titles = {}

            # ---- Summarize groups in parallel (with caching)
            with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                fut_map = {}
                for gi, group in enumerate(groups):
                    # cache key: hashed concatenation of child node contents and page-range metadata
                    key_text = "|".join([f"{self.nodes[nid].content}##{self.nodes[nid].start_page}-{self.nodes[nid].end_page}" for nid in group])
                    skey = "SUM|" + hash_text(key_text)
                    if skey in self._summary_cache:
                        summaries[gi] = self._summary_cache[skey]
                    else:
                        # prepare texts for LLM summarizer: pass child contents + short metadata hints
                        texts = [f"[p{self.nodes[n].start_page}] {self.nodes[n].content[:2000]}" for n in group]
                        fut_map[exe.submit(self.openai.summarize, texts)] = (gi, skey)
                # collect
                for fut in as_completed(list(fut_map.keys())):
                    gi, skey = fut_map[fut]
                    try:
                        summary = fut.result()
                    except Exception:
                        summary = " ".join([self.nodes[nid].content for nid in groups[gi]])[:2000]
                    summaries[gi] = summary
                    self._summary_cache[skey] = summary

            # ---- Title detection for groups (parallel)
            if prefer_title_on_parent:
                with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                    fut_map = {}
                    for gi, group in enumerate(groups):
                        key_text = "|".join([self.nodes[nid].content for nid in group])
                        tkey = "TIT|" + hash_text(key_text)
                        if tkey in self._title_cache:
                            titles[gi] = self._title_cache[tkey]
                        else:
                            texts = [self.nodes[nid].content[:1200] for nid in group]  # truncate long parts for title detection
                            fut_map[exe.submit(self.openai.detect_title, texts)] = (gi, tkey)
                    for fut in as_completed(list(fut_map.keys())):
                        gi, tkey = fut_map[fut]
                        try:
                            t = fut.result()
                        except Exception:
                            t = None
                        titles[gi] = t
                        self._title_cache[tkey] = t
            else:
                titles = {gi: None for gi in range(len(groups))}

            # ---- Create parent nodes
            next_level_ids = []
            for gi, group in enumerate(groups):
                # compute aggregated page range
                pages = [(self.nodes[nid].start_page or 999999, self.nodes[nid].end_page or -1) for nid in group]
                start_p = min(p[0] for p in pages)
                end_p = max(p[1] for p in pages)

                summary = summaries.get(gi) or " ".join([self.nodes[nid].content for nid in group])[:2000]
                title = titles.get(gi)

                # metadata: aggregate children's bounding boxes (for parent we can store envelope)
                aggregated_bboxes = []
                for nid in group:
                    bboxes = self.nodes[nid].metadata.get("bboxes", [])
                    aggregated_bboxes.extend(bboxes)
                meta = {"child_count": len(group), "aggregated_bboxes": aggregated_bboxes}

                parent_id = self._make_node(content=summary, level=current_level+1, start_page=start_p, end_page=end_p, children=group, metadata=meta)
                if title:
                    self.nodes[parent_id].title = title
                    # persist update
                    self._persist_node(self.nodes[parent_id])

                # set children parent ref and persist child updates
                for nid in group:
                    self.nodes[nid].parent = parent_id
                    try:
                        self._persist_node(self.nodes[nid])
                    except Exception:
                        pass

                next_level_ids.append(parent_id)

            current_level_ids = next_level_ids
            current_level += 1

        # single node left
        self.root_id = current_level_ids[0]
        # detect title for root if missing
        if not self.nodes[self.root_id].title:
            try:
                t = self.openai.detect_title([self.nodes[self.root_id].content])
                if t:
                    self.nodes[self.root_id].title = t
                    self._persist_node(self.nodes[self.root_id])
            except Exception:
                pass

        return self.root_id

    # --- Export methods
    def export_nested(self) -> Dict[str, Any]:
        if not self.root_id:
            raise ValueError("Tree not built")
        return self.nodes[self.root_id].to_dict(self.nodes, nested=True)

    def export_flat(self, level: int = 1) -> List[Dict[str, Any]]:
        """
        Export nodes at given level (1 = immediate children of root).
        """
        if not self.root_id:
            raise ValueError("Tree not built")
        res = []
        def collect(node_id):
            node = self.nodes[node_id]
            if node.level == level:
                res.append({
                    "title": node.title or "Untitled",
                    "node_id": node.id,
                    "start_index": node.start_page,
                    "end_index": node.end_page,
                    "summary": node.content
                })
            else:
                for ch in node.children:
                    collect(ch)
        root = self.nodes[self.root_id]
        # start collecting from root
        collect(self.root_id)
        return res

# --- Example usage (do not run in automated tests) ---
if __name__ == "__main__":
    # config
    pdf_path = "example.pdf"  # replace with your PDF
    nodes_dir = "pageindex_nodes"
    model_name = DEFAULT_MODEL
    chunk_token_size = 700
    group_size = 6

    # create clients
    openai_client = OpenAIClient(model=model_name, max_concurrent_calls=3)
    chunker = TokenChunker(model_name=model_name, chunk_token_size=chunk_token_size)
    pit = PageIndexTree(openai_client, chunker, group_size=group_size, max_workers=3, nodes_dir=nodes_dir)

    # 1) extract pages + layout
    pages_layout = extract_pages_with_layout(pdf_path)  # list of page dicts

    # 2) chunk pages into leaves
    leaf_ids = pit.pages_to_leaves(pages_layout, max_chunk_tokens=chunk_token_size)
    logger.info("Created %d leaf nodes", len(leaf_ids))

    # 3) build tree (parallel, persisted)
    root = pit.build_tree_parallel(leaf_ids, prefer_title_on_parent=True, resume=True)
    logger.info("Root: %s", root)

    # 4) export nested & flat
    nested = pit.export_nested()
    flat_top = pit.export_flat(level=1)

    # 5) save outputs
    with open("pageindex_nested.json", "w", encoding="utf-8") as f:
        json.dump(nested, f, indent=2, ensure_ascii=False)
    with open("pageindex_flat_top.json", "w", encoding="utf-8") as f:
        json.dump(flat_top, f, indent=2, ensure_ascii=False)

    logger.info("Exported nested and flat outputs.")
"""
page_index_tree_full.py

Features:
- PDF layout extraction (pdfplumber) -> pages with bounding boxes for text blocks.
- Token-aware chunking using tiktoken.
- Parallel LLM summarization and title detection using OpenAI ChatCompletion (configurable model).
- Rate-limit & retry using tenacity and concurrent.Semaphore.
- Intermediate parent node persistence to disk (JSON per node) for restartability.
- Nested export and flat-level export (MCP-like top-level).
"""

import os
import json
import time
import uuid
import math
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

import pdfplumber
import tiktoken
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

# --- Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PageIndexTree")

# --- Constants / Defaults
DEFAULT_MODEL = "gpt-4o-mini"     # change if needed
DEFAULT_MAX_TOKENS_SUMMARY = 1024 # max tokens you expect the summary to consume (for prompt budgeting)
DEFAULT_CHUNK_TOKENS = 700       # tokens per leaf chunk (adjustable)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment. Set OPENAI_API_KEY before running.")
openai.api_key = OPENAI_API_KEY

# --- helpers
def deterministic_id(prefix: str, counter: int, width: int = 4) -> str:
    return f"{prefix}{str(counter).zfill(width)}"

def hash_text(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()

# --- Node dataclass
@dataclass
class TreeNode:
    id: str
    content: str                        # summary (for internal nodes) or original chunk text (for leaf)
    level: int                          # 0 = leaf
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    title: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # includes bounding boxes for leaves

    def to_dict(self, nodes_map: Dict[str, "TreeNode"], nested: bool = True) -> Dict[str, Any]:
        base = {
            "id": self.id,
            "title": self.title,
            "summary": self.content,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "level": self.level,
            "metadata": self.metadata
        }
        if nested:
            base["children"] = [nodes_map[c].to_dict(nodes_map, nested=True) for c in self.children]
        else:
            base["children"] = self.children[:]
        return base

# --- OpenAI call wrappers with rate-limit/retry
class OpenAIClient:
    def __init__(self, model: str = DEFAULT_MODEL, max_concurrent_calls: int = 4, max_retries: int = 6):
        self.model = model
        self.semaphore = Semaphore(max_concurrent_calls)  # global concurrency limit
        self.max_retries = max_retries

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception)
    )
    def _call_chat(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> str:
        """
        Basic wrapper around openai.ChatCompletion.create with exponential backoff.
        The retry decorator will retry on exceptions (network, 429, 5xx).
        We also gate concurrent calls with a semaphore.
        """
        acquired = self.semaphore.acquire(timeout=120)
        if not acquired:
            raise TimeoutError("Could not acquire semaphore for OpenAI calls (timeout).")
        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            # Extract content
            choice = resp.choices[0]
            content = choice.message["content"] if "message" in choice else choice["text"]
            return content.strip()
        finally:
            # small sleep to avoid burst releasing causing spikes
            time.sleep(0.1)
            self.semaphore.release()

    def summarize(self, texts: List[str], prompt_template: Optional[str] = None, max_tokens: int = DEFAULT_MAX_TOKENS_SUMMARY) -> str:
        """
        Summarize a list of texts into a single short summary.
        """
        joined = "\n\n".join(texts)
        if not prompt_template:
            prompt_template = (
                "You are a concise summarization assistant. Summarize the following passages into a single "
                "short, accurate, and informative section summary (3-6 sentences). Focus on what the section is about.\n\n"
                "Passages:\n{passages}\n\nSummary:"
            )
        prompt = prompt_template.format(passages=joined)
        messages = [{"role":"user", "content": prompt}]
        try:
            return self._call_chat(messages, max_tokens=max_tokens, temperature=0.0)
        except RetryError as e:
            logger.exception("OpenAI summarize RetryError")
            # fallback: return a truncated concatenation
            return joined[:2000]

    def detect_title(self, texts: List[str], prompt_template: Optional[str] = None, max_tokens: int = 64) -> Optional[str]:
        """
        Detect concise title/heading for a group of passages.
        Returns short phrase or None.
        """
        joined = "\n\n".join(texts)
        if not prompt_template:
            prompt_template = (
                "Given the following passages that belong to the same document section, return the most likely "
                "section heading/title in 1 short phrase (no punctuation). If none, return an empty string.\n\n"
                "Passages:\n{passages}\n\nTitle:"
            )
        prompt = prompt_template.format(passages=joined)
        messages = [{"role":"user", "content": prompt}]
        try:
            out = self._call_chat(messages, max_tokens=max_tokens, temperature=0.0)
            out = out.strip()
            if not out:
                return None
            # clean common artifacts
            out = out.strip('" \n\r\t')
            # If LLM returns more than a short phrase, truncate to first line/phrase
            out = out.splitlines()[0]
            if len(out.split()) > 12:
                out = " ".join(out.split()[:12])
            return out
        except RetryError:
            return None

# --- Token-based chunker using tiktoken
class TokenChunker:
    def __init__(self, model_name: str = DEFAULT_MODEL, chunk_token_size: int = DEFAULT_CHUNK_TOKENS):
        # choose encoding that matches model family; fall back to cl100k_base
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chunk_token_size = chunk_token_size

    def tokens(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def token_len(self, text: str) -> int:
        return len(self.tokens(text))

    def chunk_text(self, text: str, max_tokens: Optional[int] = None) -> List[str]:
        """
        Split text into chunks <= max_tokens (or self.chunk_token_size) using the tokenizer.
        Returns list of string chunks.
        Strategy: attempt to split on paragraph breaks, then fall back to token slices.
        """
        if max_tokens is None:
            max_tokens = self.chunk_token_size

        # split on double newlines first (paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""
        current_tokens = 0

        for p in paragraphs:
            p_tokens = self.token_len(p)
            if current_tokens + p_tokens <= max_tokens:
                current = (current + "\n\n" + p).strip() if current else p
                current_tokens += p_tokens
            else:
                # if p itself > max_tokens, hard-split it via tokens
                if p_tokens > max_tokens:
                    # flush current
                    if current:
                        chunks.append(current)
                        current = ""
                        current_tokens = 0
                    # break p into token slices
                    tokens = self.tokens(p)
                    for i in range(0, len(tokens), max_tokens):
                        chunk_tokens = tokens[i:i+max_tokens]
                        chunk_text = self.encoding.decode(chunk_tokens)
                        chunks.append(chunk_text)
                else:
                    # flush current and start new
                    if current:
                        chunks.append(current)
                    current = p
                    current_tokens = p_tokens
        if current:
            chunks.append(current)
        # final guard: ensure no chunk exceeds max_tokens
        final_chunks = []
        for c in chunks:
            if self.token_len(c) <= max_tokens:
                final_chunks.append(c)
            else:
                # token-slice
                tokens = self.tokens(c)
                for i in range(0, len(tokens), max_tokens):
                    final_chunks.append(self.encoding.decode(tokens[i:i+max_tokens]))
        return final_chunks

# --- PDF layout extraction (pdfplumber) -> pages list with blocks and bboxes
def extract_pages_with_layout(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Returns list (page_indexed from 1) of dicts:
    {
      "page_number": int,
      "blocks": [ { "text": "...", "bbox": (x0,y0,x1,y1), "x0":..., "x1":..., "top":..., "bottom":..., "fontname":..., "size":... } , ... ],
      "raw_text": "...",  # entire page text
    }
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, pg in enumerate(pdf.pages, start=1):
            blocks = []
            # pdfplumber.extract_words gives word-level with bbox and font size maybe absent; we'll group into lines/blocks using .extract_text with layout
            try:
                words = pg.extract_words(use_text_flow=True)
                # group words into simple lines by 'top' coordinate
                lines = {}
                for w in words:
                    top_key = round(float(w.get("top", 0)), 1)
                    lines.setdefault(top_key, []).append(w)
                for top_key in sorted(lines.keys()):
                    ws = lines[top_key]
                    text = " ".join(w["text"] for w in ws)
                    # approximate bounding box as min/max of words
                    x0 = min(float(w["x0"]) for w in ws)
                    x1 = max(float(w["x1"]) for w in ws)
                    top = min(float(w.get("top", 0)) for w in ws)
                    bottom = max(float(w.get("bottom", 0)) for w in ws)
                    blocks.append({
                        "text": text,
                        "bbox": (x0, top, x1, bottom),
                        "x0": x0, "x1": x1, "top": top, "bottom": bottom,
                        "fontname": ws[0].get("fontname"),
                        "size": ws[0].get("size")
                    })
            except Exception:
                # fallback to whole page text as one block
                text = pg.extract_text() or ""
                blocks.append({
                    "text": text,
                    "bbox": (0,0,0,0),
                    "x0":0,"x1":0,"top":0,"bottom":0,"fontname":None,"size":None
                })
            pages.append({"page_number": i, "blocks": blocks, "raw_text": pg.extract_text() or ""})
    return pages

# --- PageIndexTree implementation
class PageIndexTree:
    def __init__(
        self,
        openai_client: OpenAIClient,
        chunker: TokenChunker,
        group_size: int = 6,
        max_workers: int = 6,
        nodes_dir: str = "pageindex_nodes",
        id_prefix: str = "N"
    ):
        self.openai = openai_client
        self.chunker = chunker
        self.group_size = group_size
        self.max_workers = max_workers
        self.nodes: Dict[str, TreeNode] = {}
        self.root_id: Optional[str] = None
        self._counter = 0
        self.id_prefix = id_prefix
        self.nodes_dir = nodes_dir
        os.makedirs(self.nodes_dir, exist_ok=True)
        # caches to avoid duplicate LLM calls across runs (in-memory)
        self._summary_cache = {}  # key -> summary
        self._title_cache = {}    # key -> title

    def _persist_node(self, node: TreeNode):
        path = os.path.join(self.nodes_dir, f"node_{node.id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(node), f, ensure_ascii=False, indent=2)

    def _load_persisted_nodes(self):
        """Load any nodes already saved in nodes_dir into self.nodes (useful for resume)."""
        for fname in os.listdir(self.nodes_dir):
            if not fname.startswith("node_") or not fname.endswith(".json"):
                continue
            path = os.path.join(self.nodes_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    node = TreeNode(**obj)
                    self.nodes[node.id] = node
                    # increment internal counter from numeric suffix if possible
                    try:
                        num = int(node.id.lstrip(self.id_prefix))
                        if num > self._counter:
                            self._counter = num
                    except Exception:
                        pass
            except Exception:
                logger.exception("Failed to load persisted node %s", path)

    def _make_node(self, content: str, level: int, start_page: int = None, end_page: int = None, children: List[str] = None, metadata: Dict = None) -> str:
        self._counter += 1
        node_id = deterministic_id(self.id_prefix, self._counter, width=6)
        node = TreeNode(
            id=node_id,
            content=content,
            level=level,
            children=list(children) if children else [],
            parent=None,
            title=None,
            start_page=start_page,
            end_page=end_page,
            metadata=metadata or {}
        )
        self.nodes[node_id] = node
        # persist to disk immediately for restartability
        try:
            self._persist_node(node)
        except Exception:
            logger.exception("Failed persisting node %s", node_id)
        return node_id

    # --- convert pages-with-layout into leaf nodes (token-aware, with bounding boxes)
    def pages_to_leaves(self, pages_with_layout: List[Dict[str, Any]], max_chunk_tokens: Optional[int] = None) -> List[str]:
        """
        pages_with_layout: output from extract_pages_with_layout
        For each page, group blocks into text, then chunk via token-chunker.
        Each leaf stores bounding boxes (list) and page number in metadata.
        """
        if max_chunk_tokens is None:
            max_chunk_tokens = self.chunker.chunk_token_size

        leaf_ids = []
        for page in pages_with_layout:
            pg_num = page["page_number"]
            # Build a sequence of block texts with bbox metadata
            blocks = page.get("blocks", [])
            # Simple heuristic: group blocks into a single page_text but keep block bboxes for mapping
            page_text = ""
            block_map = []
            for b in blocks:
                text = (b.get("text") or "").strip()
                if not text:
                    continue
                # keep mapping: length of added text -> bbox
                start_char = len(page_text)
                page_text += (("\n\n" + text) if page_text else text)
                end_char = len(page_text)
                block_map.append({"start": start_char, "end": end_char, "bbox": b.get("bbox"), "meta": {"x0":b.get("x0"), "x1":b.get("x1"), "top":b.get("top"), "bottom":b.get("bottom"), "size":b.get("size")}})

            if not page_text.strip():
                # create empty leaf to preserve page range
                nid = self._make_node(content="", level=0, start_page=pg_num, end_page=pg_num, metadata={"page": pg_num, "bboxes": []})
                leaf_ids.append(nid)
                continue

            # Token-based chunking on page_text
            chunks = self.chunker.chunk_text(page_text, max_tokens=max_chunk_tokens)

            # For each chunk, map subranges of chars back to bboxes by approximate char positions
            char_cursor = 0
            for chunk in chunks:
                # find where chunk occurs in page_text — we use char cursor to locate
                loc = page_text.find(chunk, char_cursor)
                if loc == -1:
                    # fallback: advance cursor and find anywhere
                    loc = page_text.find(chunk)
                if loc == -1:
                    # cannot map — keep whole page bboxes
                    bboxes = [bm["bbox"] for bm in block_map]
                else:
                    chunk_start = loc
                    chunk_end = loc + len(chunk)
                    # collect bboxes overlapping this span
                    bboxes = []
                    for bm in block_map:
                        if not (bm["end"] < chunk_start or bm["start"] > chunk_end):
                            bboxes.append(bm["bbox"])
                    if not bboxes:
                        bboxes = [bm["bbox"] for bm in block_map] or []
                    char_cursor = chunk_end

                metadata = {"page": pg_num, "bboxes": bboxes}
                nid = self._make_node(content=chunk, level=0, start_page=pg_num, end_page=pg_num, metadata=metadata)
                leaf_ids.append(nid)

        return leaf_ids

    # --- Build hierarchical tree bottom-up with parallel LLM summarization and title detection; persist parent nodes
    def build_tree_parallel(self, leaf_ids: List[str], prefer_title_on_parent: bool = True, resume: bool = True, progress: bool = False) -> str:
        """
        leaf_ids: list of leaf node IDs (level 0)
        resume: if True, will load persisted nodes and skip building nodes that already exist (safe restarts)
        """
        if resume:
            self._load_persisted_nodes()

        current_level_ids = [nid for nid in leaf_ids if self.nodes.get(nid)]  # ensure exist
        # if persisted nodes exist that are at higher levels, drop leaves that are already attached (we keep simple)
        # sort leaves by page and original creation order (deterministic)
        # we'll ascend until one root remains
        current_level = 0
        while len(current_level_ids) > 1:
            logger.info("Building level %d with %d nodes", current_level+1, len(current_level_ids))
            # create groups in order
            groups = [current_level_ids[i:i+self.group_size] for i in range(0, len(current_level_ids), self.group_size)]
            summaries = {}
            titles = {}

            # ---- Summarize groups in parallel (with caching)
            with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                fut_map = {}
                for gi, group in enumerate(groups):
                    # cache key: hashed concatenation of child node contents and page-range metadata
                    key_text = "|".join([f"{self.nodes[nid].content}##{self.nodes[nid].start_page}-{self.nodes[nid].end_page}" for nid in group])
                    skey = "SUM|" + hash_text(key_text)
                    if skey in self._summary_cache:
                        summaries[gi] = self._summary_cache[skey]
                    else:
                        # prepare texts for LLM summarizer: pass child contents + short metadata hints
                        texts = [f"[p{self.nodes[n].start_page}] {self.nodes[n].content[:2000]}" for n in group]
                        fut_map[exe.submit(self.openai.summarize, texts)] = (gi, skey)
                # collect
                for fut in as_completed(list(fut_map.keys())):
                    gi, skey = fut_map[fut]
                    try:
                        summary = fut.result()
                    except Exception:
                        summary = " ".join([self.nodes[nid].content for nid in groups[gi]])[:2000]
                    summaries[gi] = summary
                    self._summary_cache[skey] = summary

            # ---- Title detection for groups (parallel)
            if prefer_title_on_parent:
                with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                    fut_map = {}
                    for gi, group in enumerate(groups):
                        key_text = "|".join([self.nodes[nid].content for nid in group])
                        tkey = "TIT|" + hash_text(key_text)
                        if tkey in self._title_cache:
                            titles[gi] = self._title_cache[tkey]
                        else:
                            texts = [self.nodes[nid].content[:1200] for nid in group]  # truncate long parts for title detection
                            fut_map[exe.submit(self.openai.detect_title, texts)] = (gi, tkey)
                    for fut in as_completed(list(fut_map.keys())):
                        gi, tkey = fut_map[fut]
                        try:
                            t = fut.result()
                        except Exception:
                            t = None
                        titles[gi] = t
                        self._title_cache[tkey] = t
            else:
                titles = {gi: None for gi in range(len(groups))}

            # ---- Create parent nodes
            next_level_ids = []
            for gi, group in enumerate(groups):
                # compute aggregated page range
                pages = [(self.nodes[nid].start_page or 999999, self.nodes[nid].end_page or -1) for nid in group]
                start_p = min(p[0] for p in pages)
                end_p = max(p[1] for p in pages)

                summary = summaries.get(gi) or " ".join([self.nodes[nid].content for nid in group])[:2000]
                title = titles.get(gi)

                # metadata: aggregate children's bounding boxes (for parent we can store envelope)
                aggregated_bboxes = []
                for nid in group:
                    bboxes = self.nodes[nid].metadata.get("bboxes", [])
                    aggregated_bboxes.extend(bboxes)
                meta = {"child_count": len(group), "aggregated_bboxes": aggregated_bboxes}

                parent_id = self._make_node(content=summary, level=current_level+1, start_page=start_p, end_page=end_p, children=group, metadata=meta)
                if title:
                    self.nodes[parent_id].title = title
                    # persist update
                    self._persist_node(self.nodes[parent_id])

                # set children parent ref and persist child updates
                for nid in group:
                    self.nodes[nid].parent = parent_id
                    try:
                        self._persist_node(self.nodes[nid])
                    except Exception:
                        pass

                next_level_ids.append(parent_id)

            current_level_ids = next_level_ids
            current_level += 1

        # single node left
        self.root_id = current_level_ids[0]
        # detect title for root if missing
        if not self.nodes[self.root_id].title:
            try:
                t = self.openai.detect_title([self.nodes[self.root_id].content])
                if t:
                    self.nodes[self.root_id].title = t
                    self._persist_node(self.nodes[self.root_id])
            except Exception:
                pass

        return self.root_id

    # --- Export methods
    def export_nested(self) -> Dict[str, Any]:
        if not self.root_id:
            raise ValueError("Tree not built")
        return self.nodes[self.root_id].to_dict(self.nodes, nested=True)

    def export_flat(self, level: int = 1) -> List[Dict[str, Any]]:
        """
        Export nodes at given level (1 = immediate children of root).
        """
        if not self.root_id:
            raise ValueError("Tree not built")
        res = []
        def collect(node_id):
            node = self.nodes[node_id]
            if node.level == level:
                res.append({
                    "title": node.title or "Untitled",
                    "node_id": node.id,
                    "start_index": node.start_page,
                    "end_index": node.end_page,
                    "summary": node.content
                })
            else:
                for ch in node.children:
                    collect(ch)
        root = self.nodes[self.root_id]
        # start collecting from root
        collect(self.root_id)
        return res

# --- Example usage (do not run in automated tests) ---
if __name__ == "__main__":
    # config
    pdf_path = "example.pdf"  # replace with your PDF
    nodes_dir = "pageindex_nodes"
    model_name = DEFAULT_MODEL
    chunk_token_size = 700
    group_size = 6

    # create clients
    openai_client = OpenAIClient(model=model_name, max_concurrent_calls=3)
    chunker = TokenChunker(model_name=model_name, chunk_token_size=chunk_token_size)
    pit = PageIndexTree(openai_client, chunker, group_size=group_size, max_workers=3, nodes_dir=nodes_dir)

    # 1) extract pages + layout
    pages_layout = extract_pages_with_layout(pdf_path)  # list of page dicts

    # 2) chunk pages into leaves
    leaf_ids = pit.pages_to_leaves(pages_layout, max_chunk_tokens=chunk_token_size)
    logger.info("Created %d leaf nodes", len(leaf_ids))

    # 3) build tree (parallel, persisted)
    root = pit.build_tree_parallel(leaf_ids, prefer_title_on_parent=True, resume=True)
    logger.info("Root: %s", root)

    # 4) export nested & flat
    nested = pit.export_nested()
    flat_top = pit.export_flat(level=1)

    # 5) save outputs
    with open("pageindex_nested.json", "w", encoding="utf-8") as f:
        json.dump(nested, f, indent=2, ensure_ascii=False)
    with open("pageindex_flat_top.json", "w", encoding="utf-8") as f:
        json.dump(flat_top, f, indent=2, ensure_ascii=False)

    logger.info("Exported nested and flat outputs.")





# ========================================================================================================

