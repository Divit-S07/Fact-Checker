# FactGuard AI — Fact-Checker & Daily News Digest

> An AI-powered fact-checking pipeline with LangGraph, Tavily search, RAG knowledge base, MCP persistence, and a premium web UI.

**Stack:** `Gemini 3.1 Flash Lite` (all LLM calls) · `Tavily` (live evidence) · `ChromaDB` (CPU/onnx vectors) · `LangGraph` · `MCP` · `FastAPI`. Runs fully on **CPU — no GPU required**.

## 🏗️ Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Pipeline                      │
│                                                     │
│  ┌──────────────┐   ┌──────────────┐               │
│  │ Claim        │──▶│ Evidence     │◀──────────┐   │
│  │ Extractor    │   │ Retriever    │           │   │
│  └──────────────┘   │ (ReAct loop) │           │   │
│                     └──────┬───────┘           │   │
│                            │                   │   │
│                     ┌──────▼───────┐           │   │
│                     │ Verdict      │           │   │
│                     │ Judge        │           │   │
│                     └──────┬───────┘           │   │
│                            │                   │   │
│                     ┌──────▼───────┐  rejected │   │
│                     │ Critic       │───────────┘   │
│                     │ (Reflection) │               │
│                     └──────┬───────┘               │
│                            │ approved              │
│                     ┌──────▼───────┐               │
│                     │ Digest       │               │
│                     │ Writer       │               │
│                     └──────────────┘               │
└─────────────────────────────────────────────────────┘
    │
    ▼
FastAPI REST API
    │
    ▼
Premium Web UI (HTML/CSS/JS)
```

## 🔧 Setup

### 1. Clone & Install
```bash
cd e:\project
python -m pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file (copy from `.env.example`):
```bash
GOOGLE_API_KEY=your_google_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get API keys from:
- **Google Gemini**: https://aistudio.google.com/apikey
- **Tavily**: https://app.tavily.com

> Lightweight by default: runs on CPU (Gemini 3.1 Flash Lite for all LLM calls, ChromaDB local vectors, no GPU required).

### 3. Run the Server
```bash
python api_server.py
```

Open **http://localhost:8000** in your browser.

Or double-click `start.bat`.

### 4. Try It
| Query (UI panel) | What happens |
|---|---|
| *"Is it true that drinking hot water cures viral infections?"* (Fact-Check) | Claim extracted → evidence from Tavily + RAG → verdict + sources |
| *Paste a forwarded message* (Fact-Check) | Splits into individual claims, checks each |
| *"Today's technology news digest"* (Daily Digest) | 5 claims from live headlines, verified, exported as Markdown |
| *"Claims I checked this week"* (History → **This Week**) | Date-filtered claim history via MCP |

---

## 📦 Project Structure

```
e:\project\
├── agent_pipeline.py      # LangGraph pipeline (5 nodes)
├── rag_knowledge_base.py  # ChromaDB RAG + trusted sources
├── mcp_server.py          # MCP server (save_digest, get_saved_claims, export_markdown)
├── mcp_client.py          # Persistent MCP stdio client used by pipeline + API
├── api_server.py          # FastAPI REST backend
├── evaluate.py            # Optional: accuracy on labelled claims (python evaluate.py)
├── requirements.txt
├── .env.example
├── data/
│   ├── chroma/            # ChromaDB vector store (CPU/onnx embeddings)
│   ├── labelled_claims.json
│   ├── evaluation_report.json
│   ├── claims.json        # Archived fact-checks (written via MCP)
│   ├── digests.json       # Saved digests (written via MCP)
│   └── exports/           # Exported Markdown reports (via MCP export_markdown)
└── frontend/
    ├── index.html         # Main UI (+ "This Week" history filter)
    ├── style.css          # Premium dark theme
    └── app.js             # Application logic
```

---

## 🧠 AI Model

All LLM calls use **`gemini-3.1-flash-lite`** only — defined once in `agent_pipeline.py:30` (`GEMINI_MODEL`).
A wrapper retries the *same* model up to 3x on transient errors (503/429/timeout); it never switches models.

Measured: ~3–6s per LLM call, ~30s per single-claim fact-check, ~2.5 min for a 5-claim digest.

---

## 🤖 Agent Nodes

| Node | Responsibility |
|------|---------------|
| **Claim Extractor** | Splits input into individual, checkable factual claims (digest mode: grounded in live Tavily headlines) |
| **Evidence Retriever** | Searches Tavily (support + contradict in parallel) + RAG archive; reruns on critic feedback |
| **Verdict Judge** | Weighs evidence; assigns True/False/Misleading/Unverifiable + confidence |
| **Critic** | Reviews for bias/weak evidence; sends back for more evidence (max 2 rounds) |
| **Digest Writer** | Compiles Markdown digest, archives verdicts to RAG, **saves via MCP `save_digest`** |

---

## 🛠️ MCP Server Tools

| Tool | Description |
|------|-------------|
| `save_digest` | Saves compiled digest with date and topic |
| `get_saved_claims` | Returns previously checked claims with filters (date range, topic, verdict) |
| `export_markdown` | Exports digest/report to a `.md` file |

All persistence runs **through MCP**: the pipeline/API call these tools via `mcp_client.py`
(a persistent stdio session to `mcp_server.py`). The Digest Writer node saves each digest itself.

---

## 🌐 REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/fact-check` | POST | Run fact-checking pipeline |
| `/api/daily-digest` | POST | Generate daily news digest |
| `/api/saved-claims` | GET | Get claim history with filters |
| `/api/digests` | GET | List all saved digests |
| `/api/exports` | GET | List exported Markdown files |
| `/api/trusted-sources` | GET | Get trusted sources registry |
| `/api/stats` | GET | Get usage statistics |

---

## ⚖️ Guardrails

- **Unverifiable** verdict when evidence is insufficient — no guessing
- **Both sides** of evidence always presented
- **Political bias guardrail** applied to all verdicts
- Every source is **cited and credibility-rated**
- Critic reflection loop reviews each verdict before publication

---

## 📊 Evaluate Verdict Accuracy (optional)

```bash
python evaluate.py      # full labelled set (12 claims)
python evaluate.py 5    # first 5 claims only
```

Runs the pipeline against `data/labelled_claims.json` (known True/False/Misleading/Unverifiable claims)
and writes `data/evaluation_report.json` with exact-match accuracy.

---

## 🔌 Run MCP Server Standalone

```bash
python mcp_server.py
```

The MCP server communicates via stdio and can be integrated with any MCP-compatible client (Claude Desktop, etc.).
