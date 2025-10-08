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

import re
from typing import List, Dict, Any

def parse_document_sections(text: str, custom_heading_keywords: List[str] = None) -> List[Dict[str, Any]]:
    """
    Parse a document into structured sections/subsections.
    Detects headings starting with numbered patterns (1., 1.1. etc.)
    and also custom keywords like 'Schedule', 'Addendum', etc.
    Ignores mid-content numeric mentions.
    """
    if custom_heading_keywords is None:
        custom_heading_keywords = []

    # Preprocess
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    numbered_heading_re = re.compile(r"^(?P<number>\d+(\.\d+)*)(?:\s*[-–—.]?\s*)(?P<title>.+)$")
    custom_heading_re = re.compile(
        r"^(?:" + "|".join([re.escape(k) for k in custom_heading_keywords]) + r")(\b|:)",
        flags=re.IGNORECASE
    )

    def detect_heading_level(line: str) -> int:
        """Return hierarchy depth (e.g. 1.1.2 -> level 3)."""
        match = numbered_heading_re.match(line)
        if match:
            return len(match.group("number").split("."))
        return 1  # default

    def is_heading(line: str) -> bool:
        """Robust heading detection."""
        # Must start with number like 1., 1.1., etc.
        if numbered_heading_re.match(line):
            # Avoid misclassifying content like "as per clause 1.2 above"
            # -> too long or has sentence-like structure
            if len(line.split()) <= 15 and not line.endswith("."):
                return True
        
        # Match known custom heading keywords like Schedule, Addendum, etc.
        if custom_heading_re.match(line):
            # Custom headings are usually short lines
            if len(line.split()) <= 12:
                return True
        
        return False

    sections = []
    stack = []

    for line in lines:
        if is_heading(line):
            level = detect_heading_level(line)
            match = numbered_heading_re.match(line)
            number = match.group("number") if match else None
            title = match.group("title").strip() if match else line.strip()

            node = {
                "number": number,
                "title": title,
                "level": level,
                "content": "",
                "subsections": []
            }

            # Manage hierarchy
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            if stack:
                stack[-1]["subsections"].append(node)
            else:
                sections.append(node)
            stack.append(node)

        else:
            if stack:
                stack[-1]["content"] += " " + line
            else:
                # handle preamble text
                if sections and "content" in sections[-1]:
                    sections[-1]["content"] += " " + line
                else:
                    sections.append({
                        "number": None,
                        "title": "Preamble",
                        "level": 0,
                        "content": line,
                        "subsections": []
                    })

    # Cleanup whitespace
    def trim_content(sections):
        for sec in sections:
            sec["content"] = sec["content"].strip()
            trim_content(sec["subsections"])
    trim_content(sections)

    return sections


# Example Usage
sample_text = """
1. Introduction
This contract outlines the terms of engagement.

1.1 Background
The company engages the supplier for project X.

Schedule A: Payment Terms
Payments will be made quarterly.

Addendum 1
Any future modifications shall be documented here.
"""

sections = parse_document_sections(
    sample_text,
    custom_heading_keywords=["Schedule", "Addendum", "Annexure", "Exhibit"]
)

import json
print(json.dumps(sections, indent=2))



# ============================================================================================================