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

# ---------------- STEP 1: CLAUSE SEGMENTER ----------------
def segment_contract_into_clauses(text):
    """
    Segments contract into sections and subclauses without breaking sentences.
    Returns a list of (section_header, clause_text).
    """
    sections = re.split(r'(?P<header>^[A-Z][A-Za-z0-9\s\.\-:]{3,})\n', text, flags=re.MULTILINE)
    
    clauses = []
    current_header = None

    for i in range(len(sections)):
        if re.match(r'^[A-Z][A-Za-z0-9\s\.\-:]{3,}$', sections[i].strip()):
            current_header = sections[i].strip()
        elif current_header and len(sections[i].strip()) > 0:
            sentences = re.split(r'(?<=[.!?])\s+', sections[i].strip())
            chunk = ""
            for sentence in sentences:
                if len(chunk + " " + sentence) > 1500:
                    clauses.append((current_header, chunk.strip()))
                    chunk = sentence
                else:
                    chunk += " " + sentence
            if chunk:
                clauses.append((current_header, chunk.strip()))
    return clauses

# ---------------- STEP 2: PARALLEL EMBEDDING ----------------
def embed_clauses_parallel(clauses):
    with ThreadPoolExecutor(max_workers=8) as executor:
        vectors = list(executor.map(lambda c: embeddings.embed_query(c[1]), clauses))
    return np.array(vectors, dtype="float32")

# ---------------- STEP 3: BUILD FAISS STORE ----------------
def build_faiss_index(clauses):
    docs = [Document(page_content=c[1], metadata={"section": c[0], "index": i}) for i, c in enumerate(clauses)]
    vectors = embed_clauses_parallel(clauses)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    faiss_store = FAISS(embedding_function=embeddings, index=index, documents=docs)
    return faiss_store

# ---------------- STEP 4: RETRIEVAL ----------------
def retrieve_supplier_clause(template_clause, template_header, faiss_store, top_k=2):
    results = faiss_store.similarity_search_with_score(template_clause, k=top_k)
    filtered = [r for r in results if template_header.lower() in r[0].metadata["section"].lower()]
    return filtered if filtered else results

# ---------------- STEP 5: DIFFERENCE HIGHLIGHTER ----------------
def highlight_differences(text1, text2):
    diff = difflib.unified_diff(text1.split(), text2.split(), lineterm="", n=0)
    return "\n".join(diff)

# ---------------- STEP 6: SYSTEM PROMPT ----------------
system_prompt = """
You are a Contract Clause Retrieval Agent. 
Your goal is to find the most relevant clause(s) in the supplier’s contract that correspond to a given template clause.
Use the clause’s section header as the primary retrieval key.
If the supplier’s clause has modifications, rewordings, or merged content, identify and highlight them.
Return clear explanations of similarities and differences.
"""

# ---------------- STEP 7: AGENT DEFINITION ----------------
def create_clause_retrieval_agent(faiss_store):
    tools = [
        Tool(
            name="RetrieveSupplierClause",
            func=lambda query: retrieve_supplier_clause(query["clause"], query["header"], faiss_store),
            description="Retrieve the supplier’s clause matching the given template clause and section header."
        ),
        Tool(
            name="CompareClauseVersions",
            func=lambda pair: highlight_differences(pair[0], pair[1]),
            description="Highlight textual differences between template and supplier clause."
        )
    ]
    
    agent = initialize_agent(
        tools,
        llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        system_message=system_prompt
    )
    return agent

# ---------------- STEP 8: DEMO ----------------
if __name__ == "__main__":
    template_text = open("template_contract.txt").read()
    supplier_text = open("supplier_contract.txt").read()

    template_clauses = segment_contract_into_clauses(template_text)
    supplier_clauses = segment_contract_into_clauses(supplier_text)

    faiss_store = build_faiss_index(supplier_clauses)
    agent = create_clause_retrieval_agent(faiss_store)

    # Example: use clause with header "Confidentiality"
    template_header, template_clause = template_clauses[5]

    print("Template Header:", template_header)
    print("Template Clause:", template_clause[:400], "...\n")

    query = {"header": template_header, "clause": template_clause}
    results = retrieve_supplier_clause(template_clause, template_header, faiss_store, top_k=2)

    print("Matched Supplier Clause:\n", results[0][0].page_content)
    print("\nDifferences:\n", highlight_differences(template_clause, results[0][0].page_content))
