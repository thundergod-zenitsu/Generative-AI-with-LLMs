import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
import difflib

# -------- CONFIG --------
openai_api_key = "YOUR_API_KEY"
llm = ChatOpenAI(model="gpt-4o", temperature=0)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# -------- STEP 1: CLAUSE SPLITTING --------
def split_into_clauses(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100, separators=["\n\n", "\n", ".", ";"]
    )
    return splitter.split_text(text)

# -------- STEP 2: PARALLEL EMBEDDING --------
def embed_clauses_parallel(clauses):
    with ThreadPoolExecutor(max_workers=8) as executor:
        vectors = list(executor.map(lambda c: embeddings.embed_query(c), clauses))
    return np.array(vectors, dtype="float32")

# -------- STEP 3: BUILD FAISS INDEX --------
def build_faiss_index(clauses):
    vectors = embed_clauses_parallel(clauses)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return FAISS(embedding_function=embeddings, index=index, documents=clauses)

# -------- STEP 4: RETRIEVAL --------
def retrieve_clause(template_clause, faiss_store, top_k=2):
    results = faiss_store.similarity_search(template_clause, k=top_k)
    return results

# -------- STEP 5: TEXT COMPARISON --------
def highlight_differences(text1, text2):
    diff = difflib.unified_diff(
        text1.split(), text2.split(), lineterm="", n=0
    )
    return "\n".join(diff)

# -------- STEP 6: BUILD AGENT --------
def create_clause_retrieval_agent(faiss_store):
    tools = [
        Tool(
            name="RetrieveClause",
            func=lambda clause: retrieve_clause(clause, faiss_store),
            description="Retrieve the most similar clauses from the supplier contract"
        ),
        Tool(
            name="CompareText",
            func=lambda pair: highlight_differences(pair[0], pair[1]),
            description="Highlight differences between two clauses"
        )
    ]
    
    agent = initialize_agent(
        tools,
        llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent

# -------- STEP 7: RUN DEMO --------
if __name__ == "__main__":
    template_text = open("template_contract.txt").read()
    supplier_text = open("supplier_contract.txt").read()

    template_clauses = split_into_clauses(template_text)
    supplier_clauses = split_into_clauses(supplier_text)

    faiss_store = build_faiss_index(supplier_clauses)

    agent = create_clause_retrieval_agent(faiss_store)

    query_clause = template_clauses[3]  # e.g. "Confidentiality"
    print("Template Clause:\n", query_clause)

    results = retrieve_clause(query_clause, faiss_store, top_k=1)
    print("\nSupplier Clause Match:\n", results[0].page_content)

    diff = highlight_differences(query_clause, results[0].page_content)
    print("\nChanges Detected:\n", diff)
