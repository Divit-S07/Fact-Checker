"""
FastAPI Backend for Fact-Checker & Daily News Digest
Provides REST endpoints consumed by the frontend UI.
"""

import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agent_pipeline import run_fact_check, run_daily_digest
from rag_knowledge_base import get_rag, TRUSTED_SOURCES
import mcp_client

DATA_DIR = Path(__file__).parent / "data"
EXPORTS_DIR = DATA_DIR / "exports"
CLAIMS_FILE = DATA_DIR / "claims.json"
DIGESTS_FILE = DATA_DIR / "digests.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Fact-Checker & Daily News Digest API",
    description="AI-powered pipeline for claim verification and daily news digests",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────
class FactCheckRequest(BaseModel):
    text: str
    save_result: bool = True


class DigestRequest(BaseModel):
    topic: str = "Technology"
    export_md: bool = True


class SavedClaimsFilter(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    topic: Optional[str] = None
    verdict: Optional[str] = None
    limit: int = 20


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _load_json(path: Path, default=None):
    if default is None:
        default = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _save_result_to_db(final_state: dict, topic: str = "General") -> bool:
    """Digest persistence happens in the Digest Writer node via MCP save_digest."""
    return bool(final_state.get("digest_saved"))


def _export_markdown(title: str, content: str, filename: str = "") -> str:
    """Export via the MCP export_markdown tool."""
    result = mcp_client.export_markdown(title=title, content=content, filename=filename)
    return result.get("filepath", "")


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/fact-check")
async def fact_check_endpoint(req: FactCheckRequest):
    """Run the full fact-checking pipeline on submitted text."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        final_state = run_fact_check(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    claim_results = final_state.get("claim_results", [])
    digest_md = final_state.get("digest_markdown", "")

    digest_saved = bool(final_state.get("digest_saved"))
    export_path = None

    if digest_md:
        export_path = _export_markdown(
            title=f"Fact-Check Report — {date.today()}",
            content=digest_md
        )

    return {
        "claims": final_state.get("claims", []),
        "claim_results": claim_results,
        "digest_markdown": digest_md,
        "digest_saved": digest_saved,
        "export_path": export_path,
        "mode": "fact_check"
    }


@app.post("/api/daily-digest")
async def daily_digest_endpoint(req: DigestRequest):
    """Generate a daily verified news digest on a given topic."""
    try:
        final_state = run_daily_digest(req.topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    claim_results = final_state.get("claim_results", [])
    digest_md = final_state.get("digest_markdown", "")

    export_path = None
    if req.export_md and digest_md:
        export_path = _export_markdown(
            title=f"{req.topic} Daily Digest — {date.today()}",
            content=digest_md
        )

    return {
        "topic": req.topic,
        "claim_results": claim_results,
        "digest_markdown": digest_md,
        "digest_saved": bool(final_state.get("digest_saved")),
        "export_path": export_path,
        "mode": "digest"
    }


@app.get("/api/saved-claims")
async def get_saved_claims(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    topic: Optional[str] = None,
    verdict: Optional[str] = None,
    limit: int = 50
):
    """Return previously fact-checked claims with optional filters (via MCP get_saved_claims)."""
    try:
        data = mcp_client.get_saved_claims(
            start_date=start_date,
            end_date=end_date,
            topic=topic,
            verdict=verdict,
            limit=limit,
        )
        claims = data.get("claims", [])
    except Exception:
        claims = []
    return {"total": len(claims), "claims": claims}


@app.get("/api/digests")
async def get_digests(limit: int = 20):
    """Return list of saved digests."""
    digests_db = _load_json(DIGESTS_FILE, [])
    return {"total": len(digests_db), "digests": digests_db[-limit:]}


@app.get("/api/exports")
async def list_exports():
    """List all exported Markdown files."""
    files = []
    for f in sorted(EXPORTS_DIR.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
        files.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
        })
    return {"files": files}


@app.get("/api/exports/{filename}")
async def download_export(filename: str):
    """Download a specific exported Markdown file."""
    filepath = EXPORTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=str(filepath),
        media_type="text/markdown",
        filename=filename
    )


@app.get("/api/trusted-sources")
async def get_trusted_sources():
    """Return the trusted sources registry."""
    return {"sources": TRUSTED_SOURCES}


@app.get("/api/stats")
async def get_stats():
    """Return overall statistics."""
    claims_db = _load_json(CLAIMS_FILE, [])
    digests_db = _load_json(DIGESTS_FILE, [])

    verdict_counts = {}
    topic_counts = {}
    for c in claims_db:
        v = c.get("verdict", "Unverifiable")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        t = c.get("topic", "General")
        topic_counts[t] = topic_counts.get(t, 0) + 1

    return {
        "total_claims_checked": len(claims_db),
        "total_digests": len(digests_db),
        "verdict_distribution": verdict_counts,
        "topic_distribution": topic_counts,
        "exports_count": len(list(EXPORTS_DIR.glob("*.md")))
    }


# ─────────────────────────────────────────────────────────────
# Serve Frontend
# ─────────────────────────────────────────────────────────────
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False, workers=1)
