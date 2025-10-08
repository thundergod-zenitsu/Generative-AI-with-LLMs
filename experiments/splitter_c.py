import re
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import difflib

# ---------------- CONFIG ----------------
openai_api_key = "YOUR_API_KEY"
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# ---------------- CLAUSE SEGMENTER ----------------
def segment_contract_into_clauses(text):
    """
    Segment contract by section headers & paragraphs, no sentence breaking.
    """
    sections = re.split(r'(?P<header>^[A-Z][A-Za-z0-9\s\.\-:]{3,})\n', text, flags=re.MULTILINE)
    clauses = []
    current_header = None

    for i in range(len(sections)):
        if re.match(r'^[A-Z][A-Za-z0-9\s\.\-:]{3,}$', sections[i].strip()):
            current_header = sections[i].strip()
        elif current_header and len(sections[i].strip()) > 0:
            paragraphs = re.split(r'\n\s*\n', sections[i].strip())
            for p in paragraphs:
                if len(p.strip()) > 0:
                    clauses.append((current_header, p.strip()))
    return clauses

# ---------------- PARALLEL EMBEDDING ----------------
def embed_clauses_parallel(clauses):
    with ThreadPoolExecutor(max_workers=8) as executor:
        vectors = list(executor.map(lambda c: embeddings.embed_query(c[1]), clauses))
    return np.array(vectors, dtype="float32")

# ---------------- BUILD FAISS INDEX ----------------
def build_faiss_index(clauses):
    docs = [Document(page_content=c[1], metadata={"section": c[0], "index": i}) for i, c in enumerate(clauses)]
    vectors = embed_clauses_parallel(clauses)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return FAISS(embedding_function=embeddings, index=index, documents=docs)

# ---------------- RETRIEVAL ----------------
def retrieve_supplier_chunks(template_clause, template_header, faiss_store, top_k=4):
    results = faiss_store.similarity_search_with_score(template_clause, k=top_k)
    filtered = [r for r in results if template_header.lower() in r[0].metadata["section"].lower()]
    return filtered if filtered else results

# ---------------- DEDUPLICATION / MERGE ----------------
def merge_and_clean_chunks(template_clause, retrieved_chunks):
    """
    Ask LLM to merge retrieved supplier chunks into one clean version.
    """
    merged_text = "\n\n".join([chunk[0].page_content for chunk in retrieved_chunks])
    
    merge_prompt = f"""
You are a Contract Clause Consolidation Assistant.
Given a template clause and multiple retrieved supplier clause chunks, your task is:
1. Merge the supplier chunks into one coherent clause.
2. Remove any redundancy or overlapping sentences.
3. Preserve the supplier’s language and meaning as much as possible.
4. Return the clean, final supplier clause.

Template Clause:
\"\"\"{template_clause}\"\"\"

Supplier Retrieved Chunks:
\"\"\"{merged_text}\"\"\"

Now produce the final merged supplier clause:
"""
    response = llm.invoke(merge_prompt)
    return response.content.strip()

# ---------------- DIFF FUNCTION ----------------
def highlight_differences(text1, text2):
    diff = difflib.unified_diff(text1.split(), text2.split(), lineterm="", n=0)
    return "\n".join(diff)

# ---------------- AGENT SYSTEM PROMPT ----------------
system_prompt = """
You are a Contract Clause Retrieval Agent.
You locate and reconstruct supplier clauses corresponding to given template clauses.
Always prioritize section header matches.
After retrieving, merge and clean redundant text using legal phrasing conventions.
"""

# ---------------- CREATE AGENT ----------------
def create_clause_retrieval_agent(faiss_store):
    tools = [
        Tool(
            name="RetrieveSupplierChunks",
            func=lambda query: retrieve_supplier_chunks(query["clause"], query["header"], faiss_store),
            description="Retrieve related supplier clause chunks."
        ),
        Tool(
            name="MergeSupplierChunks",
            func=lambda query: merge_and_clean_chunks(query["template_clause"], query["chunks"]),
            description="Merge and clean retrieved supplier chunks into a final clause."
        ),
        Tool(
            name="CompareClauseVersions",
            func=lambda pair: highlight_differences(pair[0], pair[1]),
            description="Highlight textual differences between template and supplier clause."
        )
    ]

    return initialize_agent(
        tools,
        llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        system_message=system_prompt
    )

# ---------------- DEMO ----------------
if __name__ == "__main__":
    template_text = open("template_contract.txt").read()
    supplier_text = open("supplier_contract.txt").read()

    template_clauses = segment_contract_into_clauses(template_text)
    supplier_clauses = segment_contract_into_clauses(supplier_text)

    faiss_store = build_faiss_index(supplier_clauses)
    agent = create_clause_retrieval_agent(faiss_store)

    # Example: template clause for "Confidentiality"
    template_header, template_clause = template_clauses[5]

    print("\n🔹 Template Header:", template_header)
    print("🔹 Template Clause:\n", template_clause[:400], "...\n")

    retrieved = retrieve_supplier_chunks(template_clause, template_header, faiss_store, top_k=4)
    print(f"Retrieved {len(retrieved)} supplier chunks.")

    merged_clause = merge_and_clean_chunks(template_clause, retrieved)
    print("\n✅ Final Merged Supplier Clause:\n", merged_clause)

    diff = highlight_differences(template_clause, merged_clause)
    print("\n🔍 Differences:\n", diff)





# ============================================================================================================
# Improved parser v4 with better subsection detection and robust heading splitting.
import re, json
from typing import List, Dict, Any

def parse_document_sections_v4(text: str, custom_heading_keywords: List[str] = None) -> List[Dict[str, Any]]:
    """
    Robust parser that:
      - Scans original lines and starts a new block when a valid heading is found
        (not just any mid-sentence number like "see clause 2.1 above").
      - Splits blocks correctly even when headings appear back-to-back without blank lines.
      - Reflows paragraph text to fix simple hyphenation and wrapped lines.
      - Builds hierarchical structure for numbered headings (1, 1.1, 1.1.1).
      - Creates placeholder parent nodes when subsections appear before their parents,
        and upgrades placeholders when parent heading is later encountered.
      - Supports custom headings (Schedule, Addendum, Annexure, etc.).
    """
    if custom_heading_keywords is None:
        custom_heading_keywords = []
    raw_lines = text.splitlines()
    
    # Normalized regexes
    numbered_re = re.compile(r'^\s*(?P<number>\d+(?:\.\d+)*)(?P<sep>[\.\)\-:\s]+)(?P<title>.*\S.*)?$')
    custom_re = None
    if custom_heading_keywords:
        custom_re = re.compile(r'^\s*(?:' + '|'.join(re.escape(k) for k in custom_heading_keywords) + r')\b[:\s-]*(?P<title>.*)$', re.I)
    
    # Helper to decide if a candidate line is a real heading (not a mid-sentence mention)
    def looks_like_heading(line_idx: int, line: str) -> bool:
        # Must match numbered or custom
        if numbered_re.match(line) is None and (custom_re is None or custom_re.match(line) is None):
            return False
        # If it's the first line in document -> heading
        if line_idx == 0:
            return True
        prev = raw_lines[line_idx - 1].rstrip()
        # If previous line is blank -> heading
        if prev.strip() == "":
            return True
        # If previous line ends with strong sentence terminator -> heading
        if prev.endswith(('.', '?', '!', ';', ':', '—', '–')):
            return True
        # If previous line itself looks like a heading -> heading (consecutive headings)
        if numbered_re.match(prev) or (custom_re is not None and custom_re.match(prev)):
            return True
        # If previous line is very short (likely a title) -> heading
        if len(prev.split()) <= 3:
            return True
        # Otherwise it's likely a mid-sentence mention -> not a heading
        return False

    # Build blocks: start new block on blank lines or when we detect a valid heading start
    blocks: List[List[str]] = []
    cur_block: List[str] = []
    for i, raw in enumerate(raw_lines):
        line = raw.rstrip()
        if line.strip() == "":
            if cur_block:
                blocks.append(cur_block)
                cur_block = []
            else:
                # multiple blank lines - ignore
                continue
        else:
            if looks_like_heading(i, line):
                # start a new block for this heading
                if cur_block:
                    blocks.append(cur_block)
                cur_block = [line.strip()]
            else:
                # append to current block (continuation)
                if not cur_block:
                    cur_block = [line.strip()]
                else:
                    cur_block.append(line.strip())
    if cur_block:
        blocks.append(cur_block)
    
    # Reflow helper
    def reflow(lines: List[str]) -> str:
        out = []
        for ln in lines:
            s = ln.strip()
            if not out:
                out.append(s)
            else:
                prev = out[-1]
                # fix hyphenation
                if prev.endswith('-'):
                    out[-1] = prev[:-1] + s.lstrip()
                # if this line starts with lowercase and prev doesn't end a sentence -> join
                elif s and s[0].islower() and not prev.endswith(('.', '?', '!', ';', ':', '—', '–', '-')):
                    out[-1] = prev + ' ' + s
                else:
                    out.append(s)
        return " ".join(out)
    
    # Data structures for nodes
    sections: List[Dict[str, Any]] = []
    number_to_node: Dict[str, Dict[str, Any]] = {}
    
    def make_placeholder(num: str, level: int):
        node = {
            "number": num,
            "title": None,
            "level": level,
            "content": "",
            "subsections": []
        }
        number_to_node[num] = node
        return node
    
    # Process each block
    for block in blocks:
        first = block[0]
        m_num = numbered_re.match(first)
        m_custom = custom_re.match(first) if custom_re else None
        block_content = reflow(block[1:]) if len(block) > 1 else ""
        
        if m_num:
            number = m_num.group('number')
            title = (m_num.group('title') or "").strip()
            level = len(number.split('.'))
            
            # Ensure parent placeholders exist if needed
            if level > 1:
                parts = number.split('.')
                for d in range(1, level):
                    parent_num = '.'.join(parts[:d])
                    if parent_num not in number_to_node:
                        # create placeholder and attach in order
                        parent_node = make_placeholder(parent_num, d)
                        if d == 1:
                            sections.append(parent_node)
                        else:
                            grandparent = '.'.join(parts[:d-1])
                            # grandparent should exist because we loop ascending
                            number_to_node[grandparent]['subsections'].append(parent_node)
            
            # Create or upgrade node for this number
            if number in number_to_node:
                node = number_to_node[number]
                node['title'] = title or node.get('title')
                node['level'] = level
            else:
                node = {
                    "number": number,
                    "title": title,
                    "level": level,
                    "content": "",
                    "subsections": []
                }
                number_to_node[number] = node
                if level == 1:
                    sections.append(node)
                else:
                    parent_num = '.'.join(number.split('.')[:level-1])
                    number_to_node[parent_num]['subsections'].append(node)
            
            # attach content
            if block_content:
                node['content'] = (node.get('content', '') + ("\n\n" + block_content if node.get('content') else block_content)).strip()
        
        elif m_custom:
            # custom heading like "Schedule A: Payment Terms"
            node = {
                "number": None,
                "title": first.strip(),
                "level": 1,
                "content": block_content,
                "subsections": []
            }
            sections.append(node)
        
        else:
            # regular paragraph: attach to last seen node (most recent by order)
            if number_to_node:
                # attach to the node with the highest numeric key that came last in sections order
                # We'll use 'last added' heuristic: track by looking at sections/subsections order.
                # Simpler: attach to the most recently created node (last in number_to_node insertion order).
                # Python 3.7+ dict preserves insertion order.
                last_key = next(reversed(number_to_node))
                node = number_to_node[last_key]
                node['content'] = (node.get('content', '') + ("\n\n" + reflow(block)) if node.get('content') else reflow(block)).strip()
            else:
                # attach to preamble
                if sections and sections[0].get('title') == 'Preamble':
                    pre = sections[0]
                else:
                    pre = {"number": None, "title": "Preamble", "level": 0, "content": "", "subsections": []}
                    sections.insert(0, pre)
                pre['content'] = (pre.get('content', '') + ("\n\n" + reflow(block)) if pre.get('content') else reflow(block)).strip()
    
    # Clean None titles and strip content
    def clean_list(lst):
        for n in lst:
            if n.get('title') is None and n.get('number'):
                n['title'] = ""  # keep empty string; caller can detect placeholder by empty title
            n['content'] = n.get('content', '').strip()
            if n.get('subsections'):
                clean_list(n['subsections'])
    clean_list(sections)
    return sections

# ---------- Test cases ----------
tests = {
    "simple_nested": """
1. Introduction
This is intro paragraph.

2. Scope
Scope text line1
2.1 Scope details
Details of scope.
2.1.1 Deep detail
Deep text.

3. General
General text here.
""",
    "headings_no_blank_lines": """1. Intro
2. Scope
2.1 Details
This is the detail paragraph.
2.2 More details
""",
    "wrapped_false_heading": """
1. Intro
This is a paragraph that was line-wrapped
and continues here. it even mentions clause 2.1 in the middle which should not be a heading
because it's continuation.

2. Following Heading
Content for the following heading.
""",
    "subsection_before_parent": """
2.1 Background
This is background but parent 2 missing yet.

2. Scope
Scope contents here.

2.2 Another Subsection
More text.
""",
    "custom_headings": """
1. Intro
Intro text.

Schedule A: Payment Terms
Payments are quarterly.

2. Main
2.1 Sub
subcontent.
"""
}

results = {}
for name, txt in tests.items():
    res = parse_document_sections_v4(txt, custom_heading_keywords=['Schedule','Addendum','Annexure','Exhibit','Appendix'])
    results[name] = res

# Print readable JSON for inspection
print(json.dumps(results, indent=2, ensure_ascii=False))






# ============================================================================================================