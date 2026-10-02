"""
MCP client for the Fact-Checker MCP server (stdio).
All persistence goes through real MCP tool calls:
  - save_digest, get_saved_claims, export_markdown

A single background task owns the stdio session (required by anyio cancel
scopes) and serves requests from a queue, so calls are fast after warm-up
and safe from both sync and async code.
"""

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

MCP_SERVER = Path(__file__).parent / "mcp_server.py"

_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_queue: Optional[asyncio.Queue] = None
_worker_task: Optional[asyncio.Future] = None
_ready = threading.Event()
_handshake = threading.Event()
_lock = threading.Lock()
_stopped = False


def _ensure_worker() -> asyncio.AbstractEventLoop:
    global _loop, _thread, _queue, _worker_task, _ready, _stopped
    with _lock:
        if _stopped:
            raise RuntimeError("MCP client is closed")
        if _loop is None or _loop.is_closed():
            _ready = threading.Event()

            def _run():
                global _loop
                _loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_loop)
                _ready.set()
                _loop.run_forever()

            _thread = threading.Thread(target=_run, daemon=True, name="mcp-client-loop")
            _thread.start()
            _ready.wait(timeout=5)

        if _queue is None or _worker_task is None or _worker_task.done():
            _queue = asyncio.Queue()
            _handshake.clear()
            _worker_task = asyncio.run_coroutine_threadsafe(_worker(), _loop)
            _handshake.wait(timeout=15)  # wait for MCP initialize to finish
    return _loop


async def _worker():
    """Owns the stdio ClientSession and serves queued tool calls."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER)],
        env=dict(os.environ),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            _handshake.set()
            while True:
                item = await _queue.get()
                if item is None:
                    break
                fut, name, arguments = item
                if fut.cancelled():
                    continue
                try:
                    result = await session.call_tool(name, arguments)
                    text = "".join(
                        getattr(c, "text", "") for c in getattr(result, "content", [])
                    )
                    try:
                        fut.set_result(json.loads(text))
                    except Exception:
                        fut.set_result({"raw": text})
                except Exception as exc:  # noqa: BLE001
                    try:
                        fut.set_exception(exc)
                    except Exception:
                        pass


async def _submit(name: str, arguments: dict):
    fut = asyncio.get_running_loop().create_future()
    await _queue.put((fut, name, arguments))
    return await fut


def call_tool(name: str, arguments: dict, timeout: float = 60.0) -> dict:
    """Thread-safe MCP tool call (works from sync or async code)."""
    loop = _ensure_worker()
    task = asyncio.run_coroutine_threadsafe(_submit(name, arguments), loop)
    try:
        return task.result(timeout=timeout)
    except Exception:
        # Worker/session likely died — force a fresh session and retry once
        global _worker_task, _queue
        with _lock:
            _worker_task = None
            _queue = None
        loop = _ensure_worker()
        task = asyncio.run_coroutine_threadsafe(_submit(name, arguments), loop)
        return task.result(timeout=timeout)


# ─────────────────────────────────────────────────────────────
# Convenience wrappers (real MCP tool invocations)
# ─────────────────────────────────────────────────────────────
def save_digest(date: str, topic: str, claims: list, digest_summary: str = "") -> dict:
    return call_tool("save_digest", {
        "date": date,
        "topic": topic,
        "claims": claims,
        "digest_summary": digest_summary,
    })


def get_saved_claims(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    topic: Optional[str] = None,
    verdict: Optional[str] = None,
    limit: int = 50,
) -> dict:
    args: dict[str, Any] = {"limit": limit}
    if start_date:
        args["start_date"] = start_date
    if end_date:
        args["end_date"] = end_date
    if topic:
        args["topic"] = topic
    if verdict:
        args["verdict"] = verdict
    return call_tool("get_saved_claims", args)


def export_markdown(title: str, content: str, filename: str = "") -> dict:
    args: dict[str, Any] = {"title": title, "content": content}
    if filename:
        args["filename"] = filename
    return call_tool("export_markdown", args)


def close():
    global _stopped, _worker_task, _queue
    _stopped = True
    if _loop is not None and not _loop.is_closed() and _queue is not None:
        try:
            fut = asyncio.run_coroutine_threadsafe(_submit_shutdown(), _loop)
            fut.result(timeout=10)
        except Exception:
            pass
        _loop.call_soon_threadsafe(_loop.stop)


async def _submit_shutdown():
    try:
        await _queue.put(None)
    except Exception:
        pass
