"""
MCP Server for Fact-Checker & Daily News Digest
Provides tools: save_digest, get_saved_claims, export_markdown
"""

import json
import os
import re
from datetime import datetime, date
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Storage paths
DATA_DIR = Path(__file__).parent / "data"
DIGESTS_FILE = DATA_DIR / "digests.json"
CLAIMS_FILE = DATA_DIR / "claims.json"
EXPORTS_DIR = DATA_DIR / "exports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Any = None):
    if default is None:
        default = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path: Path, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


server = Server("factchecker-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="save_digest",
            description=(
                "Saves a compiled daily digest with date, topic, and verified claims. "
                "Use after the Digest Writer node has assembled all verdicts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "ISO date string e.g. 2025-01-15"
                    },
                    "topic": {
                        "type": "string",
                        "description": "Topic or category of the digest e.g. 'Technology', 'Health'"
                    },
                    "claims": {
                        "type": "array",
                        "description": "List of fact-checked claim objects with verdict and sources",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {"type": "string"},
                                "verdict": {"type": "string"},
                                "confidence": {"type": "number"},
                                "summary": {"type": "string"},
                                "sources": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    },
                    "digest_summary": {
                        "type": "string",
                        "description": "Overall summary of the digest"
                    }
                },
                "required": ["date", "topic", "claims"]
            }
        ),
        Tool(
            name="get_saved_claims",
            description=(
                "Returns previously fact-checked claims and their verdicts. "
                "Can filter by date range, topic, or verdict type."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start of date range (ISO format), optional"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End of date range (ISO format), optional"
                    },
                    "topic": {
                        "type": "string",
                        "description": "Filter by topic, optional"
                    },
                    "verdict": {
                        "type": "string",
                        "enum": ["True", "False", "Misleading", "Unverifiable"],
                        "description": "Filter by verdict type, optional"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of claims to return (default 20)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="export_markdown",
            description=(
                "Exports a digest or fact-check report to a Markdown file. "
                "Returns the filepath of the exported document."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the report"
                    },
                    "content": {
                        "type": "string",
                        "description": "Full Markdown content to export"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional filename (without extension). Auto-generated if omitted."
                    }
                },
                "required": ["title", "content"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "save_digest":
        return await _save_digest(arguments)
    elif name == "get_saved_claims":
        return await _get_saved_claims(arguments)
    elif name == "export_markdown":
        return await _export_markdown(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _save_digest(args: dict) -> list[TextContent]:
    digests = _load_json(DIGESTS_FILE, [])
    claims_db = _load_json(CLAIMS_FILE, [])

    digest_id = f"digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    digest_entry = {
        "id": digest_id,
        "date": args.get("date", str(date.today())),
        "topic": args.get("topic", "General"),
        "claims": args.get("claims", []),
        "digest_summary": args.get("digest_summary", ""),
        "created_at": datetime.now().isoformat()
    }

    digests.append(digest_entry)
    _save_json(DIGESTS_FILE, digests)

    # Also archive individual claims
    for claim_obj in args.get("claims", []):
        claim_entry = {
            **claim_obj,
            "digest_id": digest_id,
            "topic": args.get("topic", "General"),
            "date": args.get("date", str(date.today())),
            "archived_at": datetime.now().isoformat()
        }
        claims_db.append(claim_entry)

    _save_json(CLAIMS_FILE, claims_db)

    return [TextContent(
        type="text",
        text=json.dumps({
            "status": "success",
            "digest_id": digest_id,
            "claims_saved": len(args.get("claims", [])),
            "message": f"Digest saved with ID {digest_id}"
        })
    )]


async def _get_saved_claims(args: dict) -> list[TextContent]:
    claims_db = _load_json(CLAIMS_FILE, [])

    start_date = args.get("start_date")
    end_date = args.get("end_date")
    topic_filter = args.get("topic", "").lower()
    verdict_filter = args.get("verdict", "")
    limit = args.get("limit", 20)

    filtered = []
    for claim in claims_db:
        claim_date = claim.get("date", "")

        if start_date and claim_date < start_date:
            continue
        if end_date and claim_date > end_date:
            continue
        if topic_filter and topic_filter not in claim.get("topic", "").lower():
            continue
        if verdict_filter and claim.get("verdict", "") != verdict_filter:
            continue

        filtered.append(claim)

    filtered = filtered[-limit:]

    return [TextContent(
        type="text",
        text=json.dumps({
            "total_found": len(filtered),
            "claims": filtered
        }, default=str)
    )]


async def _export_markdown(args: dict) -> list[TextContent]:
    title = args.get("title", "Fact-Check Report")
    content = args.get("content", "")
    filename = args.get("filename", "")

    if not filename:
        safe_title = re.sub(r'[^\w\-]', '_', title)[:40]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_title}_{timestamp}"

    filepath = EXPORTS_DIR / f"{filename}.md"

    header = f"# {title}\n\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n---\n\n"
    full_content = header + content

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_content)

    return [TextContent(
        type="text",
        text=json.dumps({
            "status": "success",
            "filepath": str(filepath),
            "filename": f"{filename}.md",
            "characters_written": len(full_content),
            "message": f"Report exported to {filepath}"
        })
    )]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
