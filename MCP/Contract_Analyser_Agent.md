
# 1) Project overview — what we’re building

**Goal:** an MCP server that exposes tools, resources and prompts to let an LLM autonomously analyze contracts:

Capabilities exposed:

* Tools

  * `extract_text_from_pdf(uri)` — extract raw text page-wise.
  * `split_into_sections(text)` — heuristically or ML-based split into sections.
  * `extract_clauses(section_text)` — extract clauses (termination, indemnity, payment, confidentiality).
  * `classify_clause(clause_text)` — classify clause type and severity (risk category).
  * `score_risk(section_or_clause)` — compute numeric risk score + rationale.
  * `summarize_contract(uri, level)` — produce executive summary (low/med/high detail).
* Resources

  * Static: uploaded contract PDFs (URI `file://...`) and a `clause_ontology` DB.
  * Dynamic: company policy DB or historical contracts (from a REST endpoint / DB).
* Prompts

  * `legal_summary`, `clause_extraction`, `risk_explanation`, `redline_recommendation`
* Trace/logging for each action (auditability)
* A reference LLM client that demonstrates the autonomy loop: discover tools/resources/prompts and orchestrate steps to answer queries like:

  * “Summarize key risky clauses and provide remediation steps for Contract X.”
  * “List termination clauses with recommended redlines.”

Security & constraints:

* Server only exposes registered tools/resources; no arbitrary code execution
* Access control hooks (placeholders) for real deployments
* All calls logged with timestamp, caller id, arguments, result hash (audit trail)

---

# 2) Architecture (high-level)

```
+---------------------+         MCP(JSON-RPC / WebSocket / stdin)        +---------------------+
|   LLM Client / UI   | <----------------------------------------------> |   MCP Server (Py)   |
| (Claude/GPT client, |                                                |  - Tools            |
|  Chainlit, Word add)|                                                |  - Resources        |
+---------------------+                                                |  - Prompts          |
                                                                        +---------------------+
                                                                                |
                                                                                v
                                                +-----------------------------------------------+
                                                | Underlying Systems: PDF parsers, DBs, APIs   |
                                                | - pdfplumber / PyMuPDF                        |
                                                | - embedding store / Postgres / Mongo / S3     |
                                                | - clause classifier model (optional)          |
                                                +-----------------------------------------------+
```

Sequence (typical):

1. Client initializes, lists `tools`, `resources`, `prompts`.
2. LLM reasons and asks to `resources/read` (get PDF metadata).
3. LLM issues `call_tool("extract_text_from_pdf", {"uri": ...})`.
4. Server extracts text, returns text piecewise; LLM may call `split_into_sections`.
5. LLM calls `extract_clauses` and `classify_clause` per section.
6. LLM triggers `score_risk` for each clause and then `prompts.invoke("legal_summary", ...)` to generate final answer.
7. All steps are appended to a trace log.

---

# 3) Repo / file layout

```
contract-mcp/
├─ server/
│  ├─ app.py                 # Main FastMCP server
│  ├─ tools.py               # Tool implementations
│  ├─ resources.py           # Resource registrations
│  ├─ prompts.py             # Prompt registrations
│  ├─ utils.py               # helper functions (pdf, text, logging)
│  ├─ requirements.txt
│  └─ Dockerfile
├─ client/
│  ├─ client_example.py     # Reference autonomy loop + examples
│  └─ requirements.txt
├─ sample_data/
│  ├─ example_contract.pdf
│  └─ clause_ontology.json
└─ README.md
```

---

# 4) Server — full code (runnable)

Below is a single-file server implementation split into sections for clarity. It uses `fastmcp` as an illustrative MCP library. If `fastmcp` is not available, the structure still applies — you can implement JSON-RPC handlers directly.

**Install dependencies** (example):

```bash
# server/requirements.txt
fastmcp>=0.1.0
pdfplumber
pydantic
python-magic
uvicorn
python-dotenv
python-multipart
requests
tqdm
```

**server/app.py** (complete — save under `server/app.py`):

```python
"""
MCP Server: Contract Analysis
Run: python app.py
Port: runs via stdin/stdout or websocket depending on FastMCP config.
"""

from fastmcp import FastMCP, Tool, Resource, Prompt
from typing import Tuple, List, Dict, Any
from pydantic import BaseModel
import pdfplumber
import os
import json
import uuid
import time
from pathlib import Path
import logging

# -----------------------
# Basic logging & tracing
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("contract-mcp")

def trace_event(action: str, details: dict):
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "action": action,
        "details": details
    }
    logger.info("TRACE %s", json.dumps(event))
    return event["id"]

# -----------------------
# Helper utils
# -----------------------
def extract_text_from_pdf_path(path: str) -> List[str]:
    """Return list of page texts."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception as e:
                pages.append("")
    return pages

def naive_section_split(text: str) -> List[Dict[str, Any]]:
    """
    Heuristic: split into sections at lines that look like headings (ALL CAPS, or numbered).
    Returns list of {id, title, content}
    """
    lines = text.splitlines()
    sections = []
    cur_title = "Preamble"
    cur_text = []
    for line in lines:
        ln = line.strip()
        # heading heuristics
        if (len(ln) > 0 and (ln.isupper() or ln.startswith("SECTION ") or ln[:3].isdigit())):
            if cur_text:
                sections.append({"id": str(uuid.uuid4()), "title": cur_title, "content": "\n".join(cur_text)})
            cur_title = ln
            cur_text = []
        else:
            cur_text.append(line)
    if cur_text:
        sections.append({"id": str(uuid.uuid4()), "title": cur_title, "content": "\n".join(cur_text)})
    return sections

def simple_clause_extractor(section_text: str) -> List[str]:
    """
    Very simple clause extractor: split by paragraph breaks and keep paragraphs longer than 40 chars.
    In production: replace with ML model or regex tuned for legal text.
    """
    paras = [p.strip() for p in section_text.split("\n\n") if len(p.strip()) > 40]
    return paras

# -----------------------
# MCP App
# -----------------------
app = FastMCP(name="contract_analysis", version="0.1.0", description="Contract analysis tools")

# -----------------------
# Resources
# -----------------------
SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"
CONTRACT_STORE = Path(os.environ.get("CONTRACT_STORE", SAMPLE_DIR))

@app.resource()
def list_contracts() -> Tuple[Resource, List[dict]]:
    """
    Resource: list of available contract metadata.
    """
    contracts = []
    for p in CONTRACT_STORE.glob("*.pdf"):
        stat = p.stat()
        contracts.append({
            "uri": f"file://{p.resolve()}",
            "name": p.name,
            "size": stat.st_size,
            "modified": stat.st_mtime
        })
    resource = Resource(
        uri="kb://contracts/list",
        name="Contracts Index",
        description="Index of uploaded contracts",
        mimeType="application/json"
    )
    return resource, contracts

@app.resource()
def clause_ontology() -> Tuple[Resource, dict]:
    """
    Resource: clause ontology (example)
    """
    ont_path = CONTRACT_STORE / "clause_ontology.json"
    if ont_path.exists():
        data = json.loads(ont_path.read_text(encoding="utf-8"))
    else:
        data = {
            "clause_types": ["termination", "indemnity", "payment", "confidentiality", "warranty", "limitation_of_liability"],
            "severity_heuristics": {"indemnity": "HIGH", "limitation_of_liability": "MEDIUM"}
        }
    res = Resource(
        uri="kb://contracts/clause_ontology",
        name="Clause Ontology",
        description="Helper ontology describing clause types and severity heuristics",
        mimeType="application/json"
    )
    return res, data

# -----------------------
# Tools
# -----------------------

class ExtractParams(BaseModel):
    uri: str

@app.tool()
def extract_text_from_pdf(params: ExtractParams) -> Dict[str, Any]:
    """
    Tool: extract_text_from_pdf
    Input: {'uri': 'file:///...'}
    Output: {'pages': [...], 'page_count': N}
    """
    trace_id = trace_event("tool.call.extract_text", {"params": params.dict()})
    uri = params.uri
    if uri.startswith("file://"):
        path = uri[len("file://"):]
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        pages = extract_text_from_pdf_path(path)
        res = {"pages": pages, "page_count": len(pages)}
        trace_event("tool.result.extract_text", {"trace_id": trace_id, "pages": len(pages)})
        return res
    else:
        # In production, implement remote fetch (e.g., S3, HTTP)
        raise ValueError("Only file:// URIs supported in this demo")

class SplitParams(BaseModel):
    text: str

@app.tool()
def split_into_sections(params: SplitParams) -> List[Dict[str, Any]]:
    trace_id = trace_event("tool.call.split_sections", {"len_text": len(params.text)})
    sections = naive_section_split(params.text)
    trace_event("tool.result.split_sections", {"trace_id": trace_id, "num_sections": len(sections)})
    return sections

class ClausesParams(BaseModel):
    section_text: str

@app.tool()
def extract_clauses(params: ClausesParams) -> List[Dict[str, Any]]:
    trace_id = trace_event("tool.call.extract_clauses", {"len_text": len(params.section_text)})
    clauses = simple_clause_extractor(params.section_text)
    out = [{"id": str(uuid.uuid4()), "text": c} for c in clauses]
    trace_event("tool.result.extract_clauses", {"trace_id": trace_id, "num_clauses": len(out)})
    return out

class ClassifyParams(BaseModel):
    clause_text: str

@app.tool()
def classify_clause(params: ClassifyParams) -> Dict[str, Any]:
    """
    Naive classifier: match keywords.
    In production: replace with an ML model / embedding classifier.
    """
    t = params.clause_text.lower()
    trace_id = trace_event("tool.call.classify_clause", {"len_text": len(t)})
    if "terminate" in t or "termination" in t:
        kind = "termination"
    elif "indemn" in t:
        kind = "indemnity"
    elif "payment" in t or "invoice" in t:
        kind = "payment"
    elif "confidential" in t:
        kind = "confidentiality"
    elif "warrant" in t:
        kind = "warranty"
    else:
        kind = "other"
    severity = "LOW"
    if kind in ("indemnity", "limitation_of_liability"):
        severity = "HIGH"
    res = {"type": kind, "severity": severity}
    trace_event("tool.result.classify_clause", {"trace_id": trace_id, "classification": res})
    return res

class RiskParams(BaseModel):
    text: str

@app.tool()
def score_risk(params: RiskParams) -> Dict[str, Any]:
    """
    Simple heuristic scoring: return numeric score 0-100 and rationale.
    """
    t = params.text.lower()
    trace_id = trace_event("tool.call.score_risk", {"len_text": len(t)})
    score = 10
    reasons = []
    if "indemn" in t:
        score += 50; reasons.append("Indemnity clause increases risk")
    if "penalty" in t or "late" in t:
        score += 20; reasons.append("Late payment/penalty")
    if "limit" in t and "liabil" in t:
        score += 30; reasons.append("Limitation of liability present")
    score = min(100, score)
    res = {"score": score, "reasons": reasons}
    trace_event("tool.result.score_risk", {"trace_id": trace_id, "score": score})
    return res

class SummarizeParams(BaseModel):
    uri: str
    level: str = "short"  # short/medium/long

@app.tool()
def summarize_contract(params: SummarizeParams) -> Dict[str, Any]:
    """
    High-level summarization by invoking extract_text -> split -> simple summarizer.
    Here we do a naive summary; replace with LLM call behind the scenes in production.
    """
    trace_id = trace_event("tool.call.summarize_contract", {"uri": params.uri, "level": params.level})
    # Extract text
    ext = extract_text_from_pdf(ExtractParams(uri=params.uri))
    combined = "\n\n".join(ext["pages"])
    # pick first N chars depending on level
    length = {"short": 300, "medium": 1200, "long": 4000}.get(params.level, 300)
    summary = combined[:length] + ("..." if len(combined) > length else "")
    res = {"summary": summary, "uri": params.uri}
    trace_event("tool.result.summarize_contract", {"trace_id": trace_id, "summary_len": len(summary)})
    return res

# -----------------------
# Prompts
# -----------------------
@app.prompt()
def legal_summary_prompt() -> Prompt:
    return Prompt(
        name="legal_summary",
        description="Summarize contract and highlight high-risks with remediation suggestions.",
        arguments=["uri", "level"],
        template=(
            "You are a legal analyst. Read the contract at {{uri}}. Produce a {{level}}-level summary, "
            "list the top 5 risky clauses with brief rationales and remediation recommendations."
        )
    )

@app.prompt()
def clause_extraction_prompt() -> Prompt:
    return Prompt(
        name="clause_extractor",
        description="Guide to extract clause text from a section",
        arguments=["section_text"],
        template=(
            "Given the section text, output a JSON array of important clauses. Each clause should be an object with keys: id, text, note."
        )
    )

# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    # If FastMCP supports run(port=...), adapt accordingly. For demo we use default run.
    print("Starting Contract Analysis MCP server...")
    app.run()
```

> Notes:
>
> * The server uses naive heuristics for clauses/classification and a simple summary. Replace those with LLM-backed prompt invocations or ML models for production.
> * Every tool call creates trace events via `trace_event(...)` for auditability.
> * Resources `list_contracts` and `clause_ontology` expose metadata useful to clients/LLMs.

---

# 5) Reference client — autonomy loop & example usage

The client demonstrates how an LLM-powered assistant (or a developer-run script) would use the MCP server. The client below *does not call any commercial LLM API directly*; it emulates an LLM deciding steps. Replace the decision logic with real LLM outputs in production.

**client/client_example.py**

```python
"""
Reference client demonstrating autonomy loop.
In production, the LLM (e.g., Claude/GPT) provides actions such as "call_tool" or "read_resource".
Here we emulate that by implementing a simple planner function.
"""

import json
from typing import Any, Dict
import requests
# We'll assume we can call the MCP server via HTTP JSON-RPC if supported.
# For demo, this client imports the server module directly if running locally.

# If you're running the server in the same process (for demo), import functions.
# Otherwise, replace direct calls by JSON-RPC client calls to the server endpoint.

# For this demo, let's import server functions directly (only works if server is a module)
try:
    from server.app import extract_text_from_pdf, split_into_sections, extract_clauses, classify_clause, score_risk, summarize_contract
    server_local = True
except Exception:
    server_local = False

def simple_planner(uri: str):
    """
    Emulated LLM planner:
      1. read resource list
      2. extract text
      3. split into sections
      4. extract clauses from each section
      5. classify + score each clause
      6. assemble summary
    """
    print("Planner: Start analysis for", uri)
    # Step 1: Extract pages
    ext = extract_text_from_pdf.__wrapped__({"uri": uri}) if server_local else None
    # For direct call, above line uses the function wrapper (fastmcp attaches .__wrapped__ in some frameworks)
    if ext is None:
        raise RuntimeError("Local server calls unavailable in demo; integrate a JSON-RPC client.")
    pages = ext["pages"]
    print(f"Extracted {len(pages)} pages")
    full_text = "\n\n".join(pages)
    # Step 2: split
    sections = split_into_sections.__wrapped__({"text": full_text})
    print(f"Split into {len(sections)} sections")
    all_clauses = []
    for sec in sections:
        clauses = extract_clauses.__wrapped__({"section_text": sec["content"]})
        for c in clauses:
            cls = classify_clause.__wrapped__({"clause_text": c["text"]})
            risk = score_risk.__wrapped__({"text": c["text"]})
            all_clauses.append({
                "section_title": sec["title"],
                "clause": c,
                "classification": cls,
                "risk": risk
            })
    # Step 3: Build summary
    summary = summarize_contract.__wrapped__({"uri": uri, "level": "short"})
    result = {
        "summary": summary["summary"],
        "clauses": all_clauses
    }
    # Print top risky clauses
    sorted_by_risk = sorted(all_clauses, key=lambda x: x["risk"]["score"], reverse=True)
    print("Top risky clauses:")
    for r in sorted_by_risk[:5]:
        print("-", r["classification"], r["risk"])
    return result

if __name__ == "__main__":
    # point to sample contract
    import os
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    sample = repo_root / "sample_data" / "example_contract.pdf"
    if not sample.exists():
        print("Place a PDF at", sample)
    else:
        uri = f"file://{sample.resolve()}"
        out = simple_planner(uri)
        print(json.dumps(out, indent=2))
```

> In production replace `__wrapped__` direct calls with a real MCP JSON-RPC client that connects to the server over WebSocket or HTTP.

---

# 6) Example `sample_data` (what to include)

* `sample_data/example_contract.pdf` — any multi-page PDF for testing (use a public sample contract)
* `sample_data/clause_ontology.json` — a small JSON with clause type examples

  ```json
  {
    "clause_types": ["termination", "indemnity", "payment", "confidentiality"],
    "examples": {
      "termination": ["This Agreement may be terminated by ...", "Either party may terminate ..."],
      "indemnity": ["To the fullest extent permitted by law, the Company will indemnify ..."]
    }
  }
  ```

---

# 7) Security, privacy & production hardening

1. **Authentication & Authorization**

   * Add API keys / JWT for clients.
   * Implement per-tool ACLs (which clients allowed to call which tools).
   * Limit file URIs accessible (whitelist directories).

2. **Sandboxing**

   * Run each heavy tool in an isolated worker process/container (to avoid memory leaks or code injection).
   * Limit execution time and memory.

3. **Rate limiting & quotas**

   * Prevent runaway LLM agent loops calling heavy tools (e.g., LLM could loop forever).
   * Add counters and max plan steps.

4. **Input validation**

   * Use Pydantic (already included) to validate all tool params.

5. **Auditing & Trace**

   * Keep immutable trace logs (append-only).
   * Log user id, request id, timestamps, tool results, hashes.

6. **Sensitive data handling**

   * Redact or encrypt sensitive fields in logs.
   * Ensure PDF contents are stored securely; retention policies.

7. **LLM prompt safety**

   * Prompts should be reviewed and versioned.
   * Keep prompt templates out of user-editable space unless authorized.

---

# 8) Testing & local run instructions

1. Create virtual env and install server deps:

```bash
cd server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place a test PDF in `sample_data/example_contract.pdf`.

3. Run server:

```bash
python app.py
# or if FastMCP supports: python -m uvicorn server.app:app --reload
```

4. Run client (in separate terminal, with same venv):

```bash
cd client
python client_example.py
```

You should see an extraction, section split, clause extraction, classification and risk scoring output.

---

# 9) Example traces (what logs look like)

Each `trace_event` call writes a JSON trace to the logs, e.g.:

```json
TRACE {"id":"d1e8...","ts":169...,"action":"tool.call.extract_text","details":{"params":{"uri":"file:///.../example_contract.pdf"}}}
TRACE {"id":"d1e8...","ts":169...,"action":"tool.result.extract_text","details":{"trace_id":"d1e8...","pages":6}}
TRACE {"id":"...","ts":...,"action":"tool.call.split_sections","details":{"len_text":12345}}
TRACE {"id":"...","ts":...,"action":"tool.result.split_sections","details":{"num_sections":12}}
```

These traces let you reconstruct the entire autonomous run.

---

# 10) Next steps & optional improvements (suggested roadmap)

1. **Replace heuristics with LLM-backed tools**

   * E.g., inside `classify_clause`, call `openai`/`anthropic` to classify clause into ontology using a structured JSON response schema; this yields far better accuracy.

2. **Use embeddings + vector DB for clause matching**

   * Store clauses as embeddings (e.g., OpenAI embeddings or open-source ones) and match to a clause library.

3. **Add a Web UI**

   * Chainlit or a small React app that acts as the MCP client and visualizes traces and clause highlights in the PDF.

4. **Access controls & UI for prompts**

   * Admin UI to add / version prompts; prompt sandbox for testing.

5. **CI tests**

   * Add unit tests for each tool using pytest; add contract sample corpus for regression tests.

6. **Dockerize & Deploy**

   * Dockerfile + Kubernetes helm for autoscaling; use job queues for heavy PDF extraction.

7. **Monitoring & metrics**

   * Track tool call latencies, error rates, and LLM tokens used per session.

---

# 11) Quick design decisions I made for the demo (so you know why)

* **Keep everything local and simple** so you can run quickly and iterate. Real-world systems should decouple heavy tasks into worker queues.
* **Naive extractors/classifiers** are placeholders — they demonstrate the protocol and wiring. Replace with model or rules engines as needed.
* **Trace events** inserted on every tool call for auditability.
* **Prompts** registered server-side so LLMs can discover them — later you’ll replace template content with more carefully engineered prompts or reference LLM invocations.

---

