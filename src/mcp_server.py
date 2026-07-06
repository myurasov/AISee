# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""The AISee MCP server, served over streamable HTTP at /mcp on the API server.

A thin adapter over the REST API (the single code path): every tool maps to
consumer endpoints only, called over localhost. Nothing to install client-side:
point an MCP client at http://HOST:PORT/mcp (with the consumer bearer token when
auth is on). The tools authenticate internally with AISEE_API_TOKEN and
deliberately never pick up AISEE_ADMIN_TOKEN, so an agent connected over MCP
cannot install/uninstall/start/stop models. Media paths are resolved on this
host - the files must already exist here.
"""

import anyio.to_thread
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .client import Client

_INSTRUCTIONS = (
    "AISee gives you eyes: send image/video files with a question or an expectation and a "
    "vision-language model on this GPU host answers. Prefer `assert_visual` when you will "
    "branch on the outcome; `look` for open questions/OCR; `watch` for whole-video analysis. "
    "Query tools block until the answer is ready - a cold model can take minutes to load, so "
    "be patient or pass wait=false to `watch` and poll `get_task`. Media paths are resolved "
    "on the AISee host itself - the files must exist there. Model management is not "
    "available over MCP."
)

# stateless: every request is self-contained, so any number of agents can connect and the
# server survives restarts without session bookkeeping. Host-header (DNS-rebinding)
# checking is off: the API serves the LAN on any address and access is gated by the
# consumer bearer token instead.
mcp = FastMCP(
    "AISee", instructions=_INSTRUCTIONS, stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))


def _client() -> Client:
    # consumer capabilities only (never the admin token); talks to the local API,
    # so tool calls flow through the same REST path as every other client
    return Client(admin=False, autostart=False)


async def _run(fn, *args, **kw):
    # tools run on the server's own event loop and call the API over HTTP; a blocking
    # call here would deadlock the loop against itself, so hop to a worker thread
    return await anyio.to_thread.run_sync(lambda: fn(*args, **kw))


def _query(kind: str, media: list[str], params: dict, wait: bool = True) -> dict:
    c = _client()
    params = {k: v for k, v in params.items() if v is not None and v is not False}
    tid = c.submit(kind, media, params)
    if not wait:
        return {"task_id": tid, "hint": "poll get_task(task_id) until status is terminal"}
    t = c.wait(tid)
    if t["status"] != "done":
        return {"task_id": tid, "status": t["status"],
                "error": (t.get("error") or {}).get("message")}
    return {"task_id": tid, "status": "done", "result": t["result"],
            "timings": t.get("timings")}


@mcp.tool()
async def look(media: list[str], question: str, model: str | None = None,
               frames: int | None = None, fps: float | None = None, native: bool = False,
               context: str | None = None, max_tokens: int | None = None) -> dict:
    """Ask a free-form question about image/video files (OCR, descriptions, "where is X").

    media: file paths on the AISee host. Video is frame-sampled (frames/fps) unless
    native=true (video-capable models only). context: background the model cannot see in
    the pixels. Blocks until the answer is ready (a cold model may take minutes to load)."""
    return await _run(_query, "look", media,
                      {"question": question, "model": model, "frames": frames,
                       "fps": fps, "native": native, "context": context,
                       "max_tokens": max_tokens})


@mcp.tool()
async def assert_visual(media: list[str], expectation: str, model: str | None = None,
                        frames: int | None = None, fps: float | None = None,
                        native: bool = False, context: str | None = None,
                        max_tokens: int | None = None) -> dict:
    """Verify an expectation about image/video files; returns {pass, reason, evidence}.

    Prefer this over `look` whenever you will branch on the outcome (tests, gates).
    Blocks until the verdict is ready."""
    return await _run(_query, "assert", media,
                      {"expectation": expectation, "model": model,
                       "frames": frames, "fps": fps, "native": native,
                       "context": context, "max_tokens": max_tokens})


@mcp.tool()
async def watch(video: str, question: str | None = None, expectation: str | None = None,
                model: str | None = None, fps: float | None = None,
                chunk_seconds: float | None = None, context: str | None = None,
                max_tokens: int | None = None, wait: bool = True) -> dict:
    """Analyze a whole video chunk by chunk (use for videos longer than ~1 minute).

    Give exactly one of question (returns per-chunk findings + a synthesized answer) or
    expectation (returns {pass, failing_ranges} with the time spans where it broke).
    fps sets temporal resolution (1 for overviews, 8-15 to hunt flicker). Long videos take
    minutes: pass wait=false to get a task_id immediately and poll get_task."""
    if bool(question) == bool(expectation):
        return {"error": "give exactly one of question / expectation"}
    return await _run(_query, "watch", [video],
                      {"question": question, "expectation": expectation,
                       "model": model, "fps": fps, "chunk_seconds": chunk_seconds,
                       "context": context, "max_tokens": max_tokens}, wait=wait)


@mcp.tool()
async def list_models() -> list[dict]:
    """List installed models with live state (running/installed/loading) and defaults."""
    return await _run(lambda: _client().models())


@mcp.tool()
async def list_tasks(status: str | None = None, model: str | None = None) -> list[dict]:
    """List tasks, newest first (filters: status, model)."""
    return await _run(lambda: _client().tasks(status=status, model=model))


@mcp.tool()
async def get_task(task_id: str) -> dict:
    """Fetch one task: status, progress (step/detail/chunk), timings, result."""
    return await _run(lambda: _client().task(task_id))


@mcp.tool()
async def cancel_task(task_id: str) -> dict:
    """Cancel a queued or running task."""
    return await _run(lambda: _client().cancel(task_id))


@mcp.tool()
async def describe() -> str:
    """The MCP tool guide for this AISee host: tool usage, behavior to plan around, and the
    installed models with strengths/weaknesses/pitfalls. Read this before choosing a model."""
    return await _run(lambda: _client().describe(flavor="mcp"))


@mcp.tool()
async def health() -> dict:
    """API liveness + per-model state summary."""
    return await _run(lambda: _client().health())


def http_app():
    """The streamable-HTTP ASGI app serving /mcp internally; the API server mounts it at
    the root (so there is no /mcp -> /mcp/ redirect) and must run
    mcp.session_manager.run() in its lifespan."""
    return mcp.streamable_http_app()
