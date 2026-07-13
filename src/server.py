# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""The AISee REST API server (FastAPI). Owns the core; the CLI is just one of its clients.
Also serves the MCP server at /mcp (streamable HTTP) as a thin adapter over the same API."""

import json
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.datastructures import UploadFile

from . import (__version__, blobs, catalog, config, creds, describe, media, mcp_server,
               paths, registry)
from .tasks import Core

OPEN_PATHS = {"/", "/v1/describe", "/v1/health", "/openapi.json", "/docs", "/redoc"}


def is_admin_route(method: str, path: str) -> bool:
    """Management/lifecycle actions: model install/uninstall/start/stop."""
    if path == "/v1/models" and method == "POST":
        return True
    if path.startswith("/v1/models/") and method in ("POST", "DELETE"):
        return True
    return False


def check_auth(method: str, path: str, bearer: str) -> tuple[int, str] | None:
    """Two-tier auth. Returns None if allowed, else (status, detail).

    - AISEE_API_TOKEN (consumer): guards query/read endpoints when set.
    - AISEE_ADMIN_TOKEN (admin): guards management endpoints when set; also
      accepted everywhere the consumer token is. With only the consumer token
      set, it guards everything (single-token mode).
    """
    if path in OPEN_PATHS or method == "OPTIONS":
        return None
    store = creds.load_store()
    consumer = os.environ.get("AISEE_API_TOKEN") or store.get("AISEE_API_TOKEN")
    admin = os.environ.get("AISEE_ADMIN_TOKEN") or store.get("AISEE_ADMIN_TOKEN")
    if is_admin_route(method, path):
        if admin:
            if bearer == admin:
                return None
            if consumer and bearer == consumer:
                return (403, "forbidden: this action requires the admin token")
            return (401, "unauthorized")
        required = consumer  # single-token mode
    else:
        required = consumer or None
    if required and bearer not in {t for t in (consumer, admin) if t}:
        return (401, "unauthorized")
    return None


def create_app() -> FastAPI:
    core = Core()
    core.start_background()

    @asynccontextmanager
    async def lifespan(app):
        # the mounted MCP transport needs its session manager running
        async with mcp_server.mcp.session_manager.run():
            yield

    app = FastAPI(title="AISee", version=__version__,
                  description="AISee is a tool that gives AI agents eyes.",
                  lifespan=lifespan)
    app.state.core = core
    # the mini console is a static file that also works opened from disk; CORS lets it
    # (and any browser client) call the API cross-origin - auth still applies
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.middleware("http")
    async def auth(request: Request, call_next):
        got = request.headers.get("authorization", "")
        bearer = got[7:].strip() if got.startswith("Bearer ") else ""
        if not bearer:  # <img>/<a> tags cannot send headers; allow ?token= as a fallback
            bearer = request.query_params.get("token", "")
        denied = check_auth(request.method, request.url.path, bearer)
        if denied:
            return JSONResponse({"detail": denied[1]}, status_code=denied[0])
        return await call_next(request)

    @app.get("/", include_in_schema=False)
    def console():
        page = Path(__file__).resolve().parent.parent / "res" / "index.html"
        if page.exists():
            return HTMLResponse(page.read_text())
        return HTMLResponse("<h1>AISee</h1><p>console file missing; see /v1/describe</p>")

    @app.get("/v1/health")
    def health():
        models = {e["slug"]: core.model_state(e["slug"]) for e in registry.list_installed()}
        return {"ok": True, "version": __version__, "models": models}

    @app.get("/v1/describe")
    def describe_api(request: Request, format: str = "markdown", flavor: str = "api"):
        """flavor=api (REST guide, default) or flavor=mcp (MCP tool guide)."""
        if format == "json":
            return describe.as_json(core)
        md = describe.as_markdown(core, flavor)
        if flavor == "mcp":
            # concrete access details for the upload recipe. The URL is not a secret; the
            # consumer token is echoed only to callers who already presented it (the MCP
            # path always does - /mcp is consumer-guarded), never on the open endpoint.
            cfg = config.load()["api"]
            host = cfg["host"] if cfg["host"] != "0.0.0.0" else (config.lan_ip() or "<host-ip>")
            md = md.replace("{{api_base}}", f"http://{host}:{cfg['port']}")
            store = creds.load_store()
            consumer = os.environ.get("AISEE_API_TOKEN") or store.get("AISEE_API_TOKEN")
            admin = os.environ.get("AISEE_ADMIN_TOKEN") or store.get("AISEE_ADMIN_TOKEN")
            got = request.headers.get("authorization", "")
            bearer = got[7:].strip() if got.startswith("Bearer ") else ""
            if not consumer:
                token_text = "(none - auth is disabled on this host)"
            elif bearer in {t for t in (consumer, admin) if t}:
                token_text = consumer
            else:
                token_text = "<consumer token - ask the host operator>"
            md = md.replace("{{consumer_token}}", token_text)
        return Response(md, media_type="text/markdown")

    @app.get("/v1/gpu")
    def gpu():
        """Live GPU stats via nvidia-smi (fields that report [N/A] come back as null)."""
        import subprocess
        fields = ("index,name,utilization.gpu,memory.used,memory.total,"
                  "power.draw,power.limit,temperature.gpu,clocks.sm,clocks.max.sm")
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            raise HTTPException(503, "nvidia-smi not available on this host")
        gpus = []
        for line in out.splitlines():
            parts = [s.strip() for s in line.split(",")]
            if len(parts) < 10:
                continue
            def num(s):
                try:
                    return float(s)
                except ValueError:
                    return None  # "[N/A]" (e.g. memory.total on GB10)
            g = {
                "index": int(parts[0]), "name": parts[1],
                "utilization_pct": num(parts[2]),
                "memory_used_mib": num(parts[3]), "memory_total_mib": num(parts[4]),
                "power_draw_w": num(parts[5]), "power_limit_w": num(parts[6]),
                "temperature_c": num(parts[7]),
                "clock_sm_mhz": num(parts[8]), "clock_sm_max_mhz": num(parts[9]),
                "memory_source": "gpu",
            }
            if g["memory_total_mib"] is None:
                # unified memory (GB10 class): nvidia-smi reports [N/A]; the GPU pool IS
                # system RAM, so report it from /proc/meminfo instead
                try:
                    mi = {line.split(":")[0]: float(line.split()[1])
                          for line in open("/proc/meminfo") if ":" in line}
                    g["memory_total_mib"] = round(mi["MemTotal"] / 1024, 1)
                    g["memory_used_mib"] = round((mi["MemTotal"] - mi["MemAvailable"]) / 1024, 1)
                    g["memory_source"] = "system-unified"
                except (OSError, KeyError, ValueError, IndexError):
                    pass
            gpus.append(g)
        return {"gpus": gpus}

    @app.get("/v1/models")
    def models():
        return [core.model_view(e) for e in registry.list_installed()]

    @app.post("/v1/models/{slug}/start")
    def model_start(slug: str):
        if not registry.get(slug):
            raise HTTPException(404, f"model '{slug}' is not installed")
        try:
            core.start_model_async(slug)
        except RuntimeError as e:  # would oversubscribe the GPU
            raise HTTPException(409, str(e))
        return {"slug": slug, "state": core.model_state(slug)}

    @app.post("/v1/models/{slug}/stop")
    def model_stop(slug: str):
        if not registry.get(slug):
            raise HTTPException(404, f"model '{slug}' is not installed")
        core.stop_model(slug)
        return {"slug": slug, "state": core.model_state(slug)}

    @app.get("/v1/config")
    def get_config():
        """Effective global configuration (config.toml merged over defaults). No secrets."""
        return config.load()

    @app.get("/v1/catalog")
    def catalog_list():
        """Built-in model catalog with installed flags (for install UIs)."""
        installed = {e["slug"] for e in registry.list_installed()}
        return [{"slug": s, "hf_id": e["hf_id"], "installed": s in installed,
                 "supports_native_video": e.get("supports_native_video", True),
                 "strengths": e.get("strengths", "")}
                for s, e in catalog.CATALOG.items()]

    @app.post("/v1/models")
    async def model_install(request: Request):
        """Install a model into the registry (catalog slug or HF id). Does not start it."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(400, "JSON body required")
        name = (body or {}).get("name", "").strip()
        if not name:
            raise HTTPException(400, "name required (catalog slug or org/Model HF id)")
        try:
            entry = registry.install(
                name, image=body.get("image"), gpu_frac=body.get("gpu_frac"),
                port=body.get("port"), idle_timeout=body.get("idle_timeout"),
                max_model_len=body.get("max_model_len"), concurrency=body.get("concurrency"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return core.model_view(entry)

    @app.delete("/v1/models/{slug}")
    def model_remove(slug: str):
        """Stop and remove a model from the registry (weights stay in the shared cache)."""
        if not registry.get(slug):
            raise HTTPException(404, f"model '{slug}' is not installed")
        if core.store.open_count(slug):
            raise HTTPException(409, "model has queued or running tasks")
        core.stop_model(slug)
        registry.remove(slug)
        return {"slug": slug, "removed": True}

    @app.get("/v1/blobs/{sha}")
    def blob_probe(sha: str):
        """Dedup probe: is this content (sha256 of the file bytes) already on the server?"""
        p = blobs.find(sha)
        return {"sha256": sha.lower(), "exists": bool(p),
                "size": p.stat().st_size if p else None}

    @app.post("/v1/blobs")
    async def blob_upload(request: Request):
        """Upload media into the content-addressed store (multipart field 'files').
        Returns the sha256 per file; reference it in tasks as 'sha256:<hash>'."""
        if not request.headers.get("content-type", "").startswith("multipart/"):
            raise HTTPException(400, "multipart/form-data required (field 'files')")
        form = await request.form()
        files = [v for v in form.getlist("files") if isinstance(v, UploadFile)]
        if not files:
            raise HTTPException(400, "no files uploaded (multipart field 'files')")
        out = []
        for f in files:
            data = await f.read()
            sha, _ = blobs.put_bytes(data, f.filename or "upload.bin")
            out.append({"sha256": sha, "size": len(data), "filename": f.filename})
        return out

    @app.post("/v1/tasks")
    async def submit(request: Request):
        """Multipart (files[] + params JSON field) or JSON with media_paths on this host.
        Media entries may be 'sha256:<hash>' references to already-uploaded blobs."""
        import uuid
        ctype = request.headers.get("content-type", "")
        tid_dir = paths.media_dir() / uuid.uuid4().hex[:12]

        def resolve_blob(ref: str) -> str:
            b = blobs.find(ref[7:])
            if not b:
                raise HTTPException(400, f"unknown blob {ref} - upload it first "
                                         "(POST /v1/blobs) or send the file")
            return str(blobs.link_into(b, tid_dir / "in"))

        if ctype.startswith("multipart/"):
            form = await request.form()
            try:
                params = json.loads(form.get("params") or "{}")
            except json.JSONDecodeError as e:
                raise HTTPException(400, f"params is not valid JSON: {e}")
            files = [v for v in form.getlist("files") if isinstance(v, UploadFile)]
            refs = params.pop("media", None)  # optional ordered refs (sha256:/filenames)
            if not files and not refs:
                raise HTTPException(400, "no files uploaded (multipart field 'files')")
            staged_by_name: dict[str, str] = {}
            staged_order: list[str] = []
            for f in files:
                data = await f.read()
                name = os.path.basename(f.filename or "upload.bin")
                # uploads flow through the blob store so identical bytes dedup next time
                _, blob = blobs.put_bytes(data, name)
                p = str(blobs.link_into(blob, tid_dir / "in", name))
                staged_by_name[name] = p
                staged_order.append(p)
            if refs is None:
                params["media"] = staged_order
            else:
                resolved = []
                for r in refs:
                    r = str(r)
                    if r.startswith("sha256:"):
                        resolved.append(resolve_blob(r))
                    elif os.path.basename(r) in staged_by_name:
                        resolved.append(staged_by_name[os.path.basename(r)])
                    else:
                        raise HTTPException(400, f"media entry '{r}' matches no uploaded file")
                params["media"] = resolved
        else:
            try:
                params = await request.json()
            except json.JSONDecodeError:
                raise HTTPException(400, "body must be JSON or multipart/form-data")
            paths_in = params.pop("media_paths", None) or params.get("media")
            if not paths_in:
                raise HTTPException(400, "media_paths required for JSON submission")
            resolved = []
            for p_in in paths_in:
                p_in = str(p_in)
                if p_in.startswith("sha256:"):
                    resolved.append(resolve_blob(p_in))
                elif os.path.exists(p_in):
                    resolved.append(p_in)
                else:
                    raise HTTPException(400, f"media not found on server host: {p_in}")
            params["media"] = resolved

        kind = params.pop("kind", None)
        if kind not in ("look", "assert", "watch"):
            raise HTTPException(400, "kind must be look | assert | watch")
        if kind == "look" and not params.get("question"):
            raise HTTPException(400, "look requires 'question'")
        if kind == "assert" and not params.get("expectation"):
            raise HTTPException(400, "assert requires 'expectation'")
        if kind == "watch" and (bool(params.get("question")) == bool(params.get("expectation"))):
            raise HTTPException(400, "watch requires exactly one of question / expectation")
        model = params.pop("model", None)
        try:
            tid = core.submit(kind, model, params)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"id": tid}

    @app.get("/v1/tasks")
    def list_tasks(status: str | None = None, model: str | None = None, limit: int = 100):
        return core.store.list_tasks(status=status, model=model, limit=limit)

    @app.get("/v1/tasks/{tid}")
    def get_task(tid: str):
        t = core.store.get(tid)
        if not t:
            raise HTTPException(404, "no such task")
        return t

    def _task_media_path(tid: str, idx: int) -> Path:
        t = core.store.get(tid)
        if not t:
            raise HTTPException(404, "no such task")
        media_list = (t.get("params") or {}).get("media") or []
        if idx < 0 or idx >= len(media_list):
            raise HTTPException(404, "no such media index")
        p = Path(media_list[idx])
        if not p.is_file():
            raise HTTPException(404, "media no longer on disk (expired?)")
        return p

    @app.get("/v1/tasks/{tid}/media/{idx}")
    def task_media(tid: str, idx: int):
        """Download one of a task's media files (index into its media list)."""
        p = _task_media_path(tid, idx)
        return FileResponse(p, filename=p.name)

    @app.get("/v1/tasks/{tid}/media/{idx}/thumb")
    def task_media_thumb(tid: str, idx: int):
        """JPEG thumbnail of a task's media (image or video first frame); cached."""
        p = _task_media_path(tid, idx)
        thumb = paths.media_dir() / tid / "thumbs" / f"{idx}.jpg"
        if not thumb.exists():
            try:
                media.thumbnail(p, thumb)
            except (RuntimeError, OSError, subprocess.CalledProcessError):
                raise HTTPException(404, "cannot thumbnail this media")
        return FileResponse(thumb, media_type="image/jpeg")

    @app.delete("/v1/tasks/{tid}")
    def cancel_task(tid: str):
        if not core.cancel(tid):
            raise HTTPException(409, "task not found or already finished")
        return {"id": tid, "canceled": True}

    # MCP over streamable HTTP at /mcp (the sub-app serves that path itself; a root mount
    # avoids a 307 on /mcp). Mounted last so it never shadows the routes above; the
    # consumer-token auth middleware applies to it like any other endpoint.
    app.mount("/", mcp_server.http_app())

    return app


def main() -> None:
    """Run the API server in the foreground (the daemon child of `aisee api start`)."""
    import uvicorn
    paths.ensure_layout()
    cfg = config.load()
    host = os.environ.get("AISEE_API_HOST", cfg["api"]["host"])
    port = int(os.environ.get("AISEE_API_PORT", cfg["api"]["port"]))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
