# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""The AISee REST API server (FastAPI). Owns the core; the CLI is just one of its clients."""

import json
import os

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from . import __version__, config, creds, describe, media, paths, registry
from .tasks import Core

OPEN_PATHS = {"/v1/describe", "/v1/health", "/openapi.json", "/docs", "/redoc"}


def create_app() -> FastAPI:
    core = Core()
    core.start_background()
    app = FastAPI(title="AISee", version=__version__,
                  description="AISee is a tool that gives AI agents eyes.")
    app.state.core = core

    @app.middleware("http")
    async def auth(request: Request, call_next):
        token = os.environ.get("AISEE_API_TOKEN") or creds.load_store().get("AISEE_API_TOKEN")
        if token and request.url.path not in OPEN_PATHS:
            got = request.headers.get("authorization", "")
            if got != f"Bearer {token}":
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/v1/health")
    def health():
        models = {e["slug"]: core.model_state(e["slug"]) for e in registry.list_installed()}
        return {"ok": True, "version": __version__, "models": models}

    @app.get("/v1/describe")
    def describe_api(format: str = "markdown"):
        if format == "json":
            return describe.as_json(core)
        return Response(describe.as_markdown(core), media_type="text/markdown")

    @app.get("/v1/models")
    def models():
        return [core.model_view(e) for e in registry.list_installed()]

    @app.post("/v1/models/{slug}/start")
    def model_start(slug: str):
        if not registry.get(slug):
            raise HTTPException(404, f"model '{slug}' is not installed")
        core.start_model_async(slug)
        return {"slug": slug, "state": core.model_state(slug)}

    @app.post("/v1/models/{slug}/stop")
    def model_stop(slug: str):
        if not registry.get(slug):
            raise HTTPException(404, f"model '{slug}' is not installed")
        core.stop_model(slug)
        return {"slug": slug, "state": core.model_state(slug)}

    @app.post("/v1/tasks")
    async def submit(request: Request):
        """Multipart (files[] + params JSON field) or JSON with media_paths on this host."""
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("multipart/"):
            form = await request.form()
            try:
                params = json.loads(form.get("params") or "{}")
            except json.JSONDecodeError as e:
                raise HTTPException(400, f"params is not valid JSON: {e}")
            files = [v for v in form.getlist("files") if isinstance(v, UploadFile)]
            if not files:
                raise HTTPException(400, "no files uploaded (multipart field 'files')")
            staged: list[str] = []
            tid_dir = None
            # stage first so the task starts with its media in place
            import uuid
            stage_id = uuid.uuid4().hex[:12]
            tid_dir = paths.media_dir() / stage_id
            for f in files:
                data = await f.read()
                staged.append(str(media.stage_bytes(data, f.filename or "upload.bin",
                                                    tid_dir / "in")))
            params["media"] = staged
        else:
            try:
                params = await request.json()
            except json.JSONDecodeError:
                raise HTTPException(400, "body must be JSON or multipart/form-data")
            paths_in = params.pop("media_paths", None) or params.get("media")
            if not paths_in:
                raise HTTPException(400, "media_paths required for JSON submission")
            missing = [p for p in paths_in if not os.path.exists(p)]
            if missing:
                raise HTTPException(400, f"media not found on server host: {missing}")
            params["media"] = list(paths_in)

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

    @app.delete("/v1/tasks/{tid}")
    def cancel_task(tid: str):
        if not core.cancel(tid):
            raise HTTPException(409, "task not found or already finished")
        return {"id": tid, "canceled": True}

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
