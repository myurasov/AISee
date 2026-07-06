# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""The AISee MCP server (stdio) - gives the calling AI agent eyes.

A thin adapter over the REST API (the single code path): every tool maps to
consumer endpoints only. It authenticates with AISEE_API_TOKEN (the consumer
token) and deliberately never picks up AISEE_ADMIN_TOKEN, so an agent connected
over MCP cannot install/uninstall/start/stop models. Run with `aisee mcp
[--server URL]`; media paths are local to this process and are uploaded.
"""

from mcp.server.fastmcp import FastMCP

from .client import Client

_SERVER: str | None = None  # set by main(); falls back to AISEE_SERVER / local config

_INSTRUCTIONS = (
    "AISee gives you eyes: send image/video files with a question or an expectation and a "
    "vision-language model on a GPU host answers. Prefer `assert_visual` when you will branch "
    "on the outcome; `look` for open questions/OCR; `watch` for whole-video analysis. Query "
    "tools block until the answer is ready - a cold model can take minutes to load, so be "
    "patient or pass wait=false to `watch` and poll `get_task`. Media paths must exist on the "
    "machine running this MCP server. Model management is not available over MCP."
)

mcp = FastMCP("AISee", instructions=_INSTRUCTIONS)


def _client() -> Client:
    # consumer capabilities only: never send the admin token, even if stored
    return Client(server=_SERVER, admin=False)


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
def look(media: list[str], question: str, model: str | None = None,
         frames: int | None = None, fps: float | None = None, native: bool = False,
         context: str | None = None, max_tokens: int | None = None) -> dict:
    """Ask a free-form question about image/video files (OCR, descriptions, "where is X").

    media: local file paths. Video is frame-sampled (frames/fps) unless native=true
    (video-capable models only). context: background the model cannot see in the pixels.
    Blocks until the answer is ready (a cold model may take minutes to load)."""
    return _query("look", media, {"question": question, "model": model, "frames": frames,
                                  "fps": fps, "native": native, "context": context,
                                  "max_tokens": max_tokens})


@mcp.tool()
def assert_visual(media: list[str], expectation: str, model: str | None = None,
                  frames: int | None = None, fps: float | None = None, native: bool = False,
                  context: str | None = None, max_tokens: int | None = None) -> dict:
    """Verify an expectation about image/video files; returns {pass, reason, evidence}.

    Prefer this over `look` whenever you will branch on the outcome (tests, gates).
    Blocks until the verdict is ready."""
    return _query("assert", media, {"expectation": expectation, "model": model,
                                    "frames": frames, "fps": fps, "native": native,
                                    "context": context, "max_tokens": max_tokens})


@mcp.tool()
def watch(video: str, question: str | None = None, expectation: str | None = None,
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
    return _query("watch", [video], {"question": question, "expectation": expectation,
                                     "model": model, "fps": fps,
                                     "chunk_seconds": chunk_seconds, "context": context,
                                     "max_tokens": max_tokens}, wait=wait)


@mcp.tool()
def list_models() -> list[dict]:
    """List installed models with live state (running/installed/loading) and defaults."""
    return _client().models()


@mcp.tool()
def list_tasks(status: str | None = None, model: str | None = None) -> list[dict]:
    """List tasks, newest first (filters: status, model)."""
    return _client().tasks(status=status, model=model)


@mcp.tool()
def get_task(task_id: str) -> dict:
    """Fetch one task: status, progress (step/detail/chunk), timings, result."""
    return _client().task(task_id)


@mcp.tool()
def cancel_task(task_id: str) -> dict:
    """Cancel a queued or running task."""
    return _client().cancel(task_id)


@mcp.tool()
def describe() -> str:
    """The server's full agent-facing guide: endpoints, task lifecycle, and the installed
    models with strengths/weaknesses/pitfalls. Read this before choosing a model."""
    return _client().describe()


@mcp.tool()
def health() -> dict:
    """API liveness + per-model state summary."""
    return _client().health()


def main(server: str | None = None) -> None:
    """Run the MCP server on stdio (the `aisee mcp` command)."""
    global _SERVER
    _SERVER = server
    try:
        _client().ensure()
    except RuntimeError as e:
        # stdout belongs to the MCP transport; complain on stderr and exit
        import sys
        print(f"aisee mcp: {e}", file=sys.stderr)
        raise SystemExit(1)
    mcp.run()


if __name__ == "__main__":
    main()
