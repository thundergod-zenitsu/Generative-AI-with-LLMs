## 🧠 1. Conceptual Overview — What DeepWiki Does

At its core, DeepWiki is a **Codebase Understanding + Documentation Generation** pipeline.

It does four things in sequence:

1. **Ingest / Parse Repository**

   * Clone a repo from GitHub (or local folder)
   * Parse files → understand the structure (modules, classes, functions, dependencies)

2. **Build Knowledge Graph**

   * Represent relationships between components
     (e.g., “`attendance_service.py` → imports `db_manager.py` → defines `mark_attendance()` used in `controller.py`”)

3. **Generate Natural-Language Summaries**

   * Use LLM(s) to summarize:

     * file-level summaries
     * class / function docstrings
     * inter-module relationships
     * architecture overview
   * Optionally visualize architecture via graph diagrams (e.g., Mermaid, Graphviz)

4. **Render as a Web App / Wiki**

   * Convert the structured summaries + diagrams into a static website (Markdown → HTML)
   * Cross-link each page (e.g., “see also this module/class”)
   * Optional: expose a **Q&A endpoint** (“Ask about this repo”) backed by retrieval-augmented generation (RAG)

---

## 🧩 2. High-Level Architecture

Here’s how you can structure a local DeepWiki clone:

```
📦 deepwiki_clone/
 ┣ 📂 backend/
 ┃ ┣ clone_repo.py
 ┃ ┣ parse_code.py
 ┃ ┣ build_graph.py
 ┃ ┣ summarize.py
 ┃ ┣ generate_docs.py
 ┃ ┗ api_server.py
 ┣ 📂 frontend/
 ┃ ┣ templates/
 ┃ ┣ static/
 ┃ ┗ app.py
 ┣ 📂 data/
 ┃ ┣ embeddings/
 ┃ ┣ summaries/
 ┃ ┗ repo_cache/
 ┗ config.yaml
```

---

## ⚙️ 3. Step-by-Step Technical Pipeline

### **Step 1: Repo Ingestion**

Use `GitPython` to clone and walk through files.

```python
from git import Repo
import os

def clone_repo(repo_url, dest_dir):
    if not os.path.exists(dest_dir):
        Repo.clone_from(repo_url, dest_dir)
    return dest_dir
```

Then filter source files by extension (`.py`, `.js`, `.java`, etc.) using `os.walk()`.

---

### **Step 2: Static Code Parsing**

Parse and analyze code structures.

For Python:

```python
import ast

def parse_python_file(file_path):
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    return {"classes": classes, "functions": functions, "imports": imports}
```

For other languages, use:

* **Tree-Sitter** (multi-language parser, bindings available in Python)
* or `universal-ctags` for structural extraction.

---

### **Step 3: Build a Repository Graph**

Construct a dependency graph between files.

```python
import networkx as nx

def build_dependency_graph(parsed_files):
    G = nx.DiGraph()
    for file, data in parsed_files.items():
        G.add_node(file, **data)
        for imp in data["imports"]:
            if imp in parsed_files:
                G.add_edge(file, imp)
    return G
```

Then you can visualize with `pyvis`, `graphviz`, or `mermaid`.

---

### **Step 4: Summarize Using LLM**

Use an LLM (e.g., OpenAI GPT-4, local model, or Ollama) to create summaries.

````python
from openai import OpenAI
client = OpenAI(api_key="YOUR_KEY")

def summarize_file(file_name, code):
    prompt = f"""
    Explain the purpose and functionality of the following Python file:
    {file_name}

    ```python
    {code}
    ```
    Provide a structured summary: Purpose, Key Classes, Key Functions, and Relationships.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
````

Store summaries in Markdown.

---

### **Step 5: Generate Documentation Pages**

Convert summaries + graph into Markdown → HTML.

Use:

* `mkdocs` or `sphinx` (if you want full documentation features)
* or your own lightweight Flask/Jinja2 template generator:

```python
from jinja2 import Environment, FileSystemLoader
import os

def generate_html_pages(parsed_data, summaries, output_dir="site"):
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("file_page.html")
    os.makedirs(output_dir, exist_ok=True)

    for file_name, summary in summaries.items():
        html = template.render(file=file_name, summary=summary, metadata=parsed_data[file_name])
        with open(f"{output_dir}/{file_name}.html", "w") as f:
            f.write(html)
```

---

### **Step 6: Add a Search / Q&A Layer (Optional)**

For DeepWiki-style interactivity:

* Store all summaries in a vector database (e.g., FAISS or Chroma)
* Use embedding model (like `text-embedding-3-large`) for retrieval

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(list(summaries.values()), embeddings)
```

Then expose a Flask API:

```python
from flask import Flask, request, jsonify

@app.route("/ask", methods=["POST"])
def ask():
    q = request.json["query"]
    docs = vector_store.similarity_search(q, k=3)
    context = "\n".join([d.page_content for d in docs])
    answer = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a code documentation assistant."},
                  {"role": "user", "content": f"Answer based on:\n{context}\n\nQ: {q}"}]
    )
    return jsonify({"answer": answer.choices[0].message.content})
```

---

### **Step 7: Serve the Wiki**

Use Flask or FastAPI to serve static docs and handle `/ask` API requests.

You can deploy locally (or even generate static HTML for offline browsing).

---

## 🧰 4. Tech Stack Summary

| Component           | Suggested Tool                      |
| ------------------- | ----------------------------------- |
| Repo Cloning        | `GitPython`                         |
| Code Parsing        | `ast`, `Tree-Sitter`, `ctags`       |
| Dependency Graph    | `networkx`, `graphviz`, `mermaid`   |
| LLM Summarization   | `OpenAI`, `Ollama`, or `LlamaIndex` |
| Embeddings & Search | `FAISS` / `ChromaDB`                |
| Web Framework       | `Flask` or `FastAPI`                |
| UI / Docs Rendering | `MkDocs`, `Jinja2`, `Bootstrap`     |
| Optional API Layer  | `/ask` endpoint with RAG backend    |

---

## 🧪 5. Optional Enhancements (To Reach DeepWiki-Level)

* 🔄 **Incremental updates** — re-generate docs only for changed files using git diffs
* 🧩 **Diagram Generation** — generate architecture diagrams via Graphviz or Mermaid
* 🔍 **Cross-linking** — automatically hyperlink between functions/classes in docs
* 🧠 **Config File Support (`.wiki.json`)** — define what parts to include/exclude
* 🕸️ **Web-based Navigation** — collapsible sidebar tree (MkDocs-like)
* 💬 **Interactive Q&A panel** — “Ask this repo” via the vector store

---






---

## 🧩 1. Updated Architecture: “DeepWiki Local”

```
📦 deepwiki_local/
 ┣ 📂 backend/
 ┃ ┣ clone_repo.py
 ┃ ┣ parse_code.py
 ┃ ┣ build_graph.py
 ┃ ┣ summarize.py
 ┃ ┣ generate_docs.py
 ┃ ┣ persist_store.py     ← NEW
 ┃ ┣ api_server.py
 ┣ 📂 data/
 ┃ ┣ repos/               ← cloned repos (each repo in its folder)
 ┃ ┣ summaries/           ← markdown summaries
 ┃ ┣ embeddings/          ← FAISS/Chroma DB files
 ┃ ┣ graphs/              ← serialized dependency graphs
 ┃ ┗ metadata.sqlite      ← SQLite DB tracking repos + metadata
 ┣ 📂 frontend/
 ┃ ┣ templates/
 ┃ ┣ static/
 ┃ ┗ app.py               ← Flask frontend
 ┗ config.yaml
```

---

## 🧠 2. Data Flow (with Persistence)

Here’s the revised flow with local persistence:

```
User uploads GitHub URL → Clone repo (data/repos/<repo_name>/)
    ↓
Parse & analyze code structure → store JSON in (data/graphs/<repo_name>.json)
    ↓
Generate summaries (LLM) → save as Markdown in (data/summaries/<repo_name>/)
    ↓
Create embeddings (FAISS/Chroma) → save to (data/embeddings/<repo_name>/)
    ↓
Generate HTML docs → save under (frontend/static/repos/<repo_name>/)
    ↓
Insert repo entry in metadata.sqlite (repo_name, path, timestamp)
    ↓
Serve via Flask dashboard — list existing repos + open wiki or query it
```

---

## 🧱 3. Database Schema (SQLite)

`metadata.sqlite` keeps everything organized:

```sql
CREATE TABLE repos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    url TEXT,
    local_path TEXT,
    summary_path TEXT,
    embedding_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Python Interface (`persist_store.py`)

```python
import sqlite3
from datetime import datetime

DB_PATH = "data/metadata.sqlite"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS repos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        url TEXT,
        local_path TEXT,
        summary_path TEXT,
        embedding_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def add_repo(name, url, local_path, summary_path, embedding_path):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO repos (name, url, local_path, summary_path, embedding_path)
    VALUES (?, ?, ?, ?, ?)
    """, (name, url, local_path, summary_path, embedding_path))
    conn.commit()
    conn.close()

def get_all_repos():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name, url, created_at FROM repos")
    rows = cur.fetchall()
    conn.close()
    return rows
```

---

## 🧮 4. Embedding Persistence

When you create embeddings for a repo, store them under a persistent path:

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

def build_vector_store(repo_name, texts):
    path = f"data/embeddings/{repo_name}/"
    os.makedirs(path, exist_ok=True)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local(path)
    return path
```

Later load it instantly:

```python
def load_vector_store(repo_name):
    path = f"data/embeddings/{repo_name}/"
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
```

---

## 🌐 5. Flask Dashboard for Persistence

```python
from flask import Flask, render_template, request, redirect, url_for
from persist_store import get_all_repos, add_repo
from backend.clone_repo import clone_repo
from backend.pipeline import process_repo  # pipeline for parse+summarize+generate

app = Flask(__name__)

@app.route('/')
def home():
    repos = get_all_repos()
    return render_template('index.html', repos=repos)

@app.route('/add', methods=['POST'])
def add_repo_route():
    repo_url = request.form['repo_url']
    name = repo_url.split('/')[-1].replace('.git', '')
    local_path = clone_repo(repo_url, f"data/repos/{name}")
    summary_path, embedding_path = process_repo(name, local_path)
    add_repo(name, repo_url, local_path, summary_path, embedding_path)
    return redirect(url_for('home'))

@app.route('/repo/<name>')
def view_repo(name):
    return render_template(f"repos/{name}/index.html")
```

---

## 🖥️ 6. Frontend Layout

### `index.html`

```html
<!DOCTYPE html>
<html>
<head><title>DeepWiki Local</title></head>
<body>
  <h1>🧠 DeepWiki Local Dashboard</h1>
  <form method="post" action="/add">
    <input type="text" name="repo_url" placeholder="https://github.com/user/repo" required>
    <button type="submit">Add Repository</button>
  </form>

  <h2>Saved Repositories</h2>
  <ul>
    {% for id, name, url, created_at in repos %}
      <li>
        <a href="/repo/{{ name }}">{{ name }}</a>
        <br><small>{{ url }} (added {{ created_at }})</small>
      </li>
    {% endfor %}
  </ul>
</body>
</html>
```

---

## 💾 7. Persistence Summary

| Data Type        | Location                             | Format          |
| ---------------- | ------------------------------------ | --------------- |
| Repo Source      | `data/repos/<repo_name>/`            | cloned Git repo |
| Summaries        | `data/summaries/<repo_name>/`        | Markdown files  |
| Graphs           | `data/graphs/<repo_name>.json`       | JSON            |
| Embeddings       | `data/embeddings/<repo_name>/`       | FAISS index     |
| Metadata         | `data/metadata.sqlite`               | SQLite          |
| Rendered Docs    | `frontend/static/repos/<repo_name>/` | HTML            |
| User-facing Data | `frontend/app.py` dashboard          | Flask app       |

---

## 🧩 8. How You’d Run It Locally

```bash
$ git clone https://github.com/yourname/deepwiki-local
$ cd deepwiki-local
$ python -m venv venv && source venv/bin/activate
$ pip install flask gitpython openai langchain chromadb jinja2 networkx faiss-cpu
$ mkdir -p data/repos data/summaries data/embeddings data/graphs
$ python -m backend.persist_store  # initializes SQLite
$ python frontend/app.py
```

Then open **[http://localhost:5000](http://localhost:5000)**
→ Paste a GitHub repo URL
→ It clones, analyzes, generates docs, saves all locally, and lists it for later.

---
