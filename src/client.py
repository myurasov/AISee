# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""HTTP client used by the CLI — the single code path to the API server."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from . import blobs, config, creds, paths


def server_url(explicit: str | None = None) -> str:
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("AISEE_SERVER")
    if env:
        return env.rstrip("/")
    return f"http://127.0.0.1:{config.load()['api']['port']}"


def is_local(url: str) -> bool:
    return "://127.0.0.1" in url or "://localhost" in url


def resolve_token(explicit: str | None = None, admin: bool = True) -> str | None:
    """Pick the bearer token: explicit > AISEE_ADMIN_TOKEN (unless admin=False) >
    AISEE_API_TOKEN, each from env then the creds store."""
    if explicit:
        return explicit
    store = creds.load_store()
    keys = ("AISEE_ADMIN_TOKEN", "AISEE_API_TOKEN") if admin else ("AISEE_API_TOKEN",)
    for k in keys:
        v = os.environ.get(k) or store.get(k)
        if v:
            return v
    return None


class Client:
    def __init__(self, server: str | None = None, autostart: bool = True,
                 token: str | None = None, admin: bool = True):
        self.base = server_url(server)
        self.autostart = autostart and is_local(self.base)
        tok = resolve_token(token, admin=admin)
        self.headers = {"Authorization": f"Bearer {tok}"} if tok else {}

    # ---------------- daemon management (bootstrap; local only) ----------------

    def api_running(self) -> bool:
        try:
            r = httpx.get(f"{self.base}/v1/health", headers=self.headers, timeout=3)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def api_start(self, wait_s: int = 30) -> bool:
        """Spawn the daemon (detached) and wait for health."""
        if self.api_running():
            return True
        paths.ensure_layout()
        log = open(paths.api_log(), "ab")
        # -P (safe path) keeps the cwd off sys.path so a directory named "aisee"
        # in the working directory can never shadow the installed package
        proc = subprocess.Popen(
            [sys.executable, "-P", "-m", "aisee.server"],
            stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            start_new_session=True, cwd=str(paths.home()))
        paths.api_pidfile().write_text(str(proc.pid))
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.api_running():
                return True
            if proc.poll() is not None:
                raise RuntimeError(f"API server exited at startup; see {paths.api_log()}")
            time.sleep(0.5)
        raise RuntimeError(f"API server did not become healthy in {wait_s}s; see {paths.api_log()}")

    def api_stop(self) -> bool:
        pidfile = paths.api_pidfile()
        if not pidfile.exists():
            return False
        try:
            pid = int(pidfile.read_text().strip())
            os.kill(pid, signal.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pidfile.unlink(missing_ok=True)
            return False
        for _ in range(20):
            try:
                os.kill(pid, 0)
                time.sleep(0.25)
            except ProcessLookupError:
                break
        pidfile.unlink(missing_ok=True)
        return True

    def ensure(self) -> None:
        if self.api_running():
            return
        if self.autostart:
            self.api_start()
            return
        raise RuntimeError(f"AISee API not reachable at {self.base} "
                           "(start it with `aisee api start`)")

    # ---------------- API calls ----------------

    def _req(self, method: str, path: str, **kw) -> httpx.Response:
        r = httpx.request(method, f"{self.base}{path}", headers=self.headers,
                          timeout=kw.pop("timeout", 30), **kw)
        if r.status_code == 401:
            raise RuntimeError("unauthorized: set AISEE_API_TOKEN (env or `aisee creds set`)")
        if r.status_code == 403:
            raise RuntimeError("forbidden: this action requires the admin token "
                               "(AISEE_ADMIN_TOKEN, env or `aisee creds set`)")
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except json.JSONDecodeError:
                detail = r.text
            raise RuntimeError(f"HTTP {r.status_code}: {detail}")
        return r

    def health(self) -> dict:
        return self._req("GET", "/v1/health").json()

    def describe(self, fmt: str = "markdown", flavor: str = "api") -> str:
        r = self._req("GET", f"/v1/describe?format={fmt}&flavor={flavor}")
        return r.text

    def models(self) -> list[dict]:
        return self._req("GET", "/v1/models").json()

    def model_start(self, slug: str) -> dict:
        return self._req("POST", f"/v1/models/{slug}/start").json()

    def model_stop(self, slug: str) -> dict:
        return self._req("POST", f"/v1/models/{slug}/stop").json()

    def blob_exists(self, sha: str) -> bool:
        return bool(self._req("GET", f"/v1/blobs/{sha}").json().get("exists"))

    def submit(self, kind: str, media_files: list[str], params: dict) -> str:
        """Submit a task; content the server already has (by sha256) is not re-uploaded.
        Media entries may also be 'sha256:<hash>' references to server-side blobs."""
        params = {"kind": kind, **params}
        refs: list[str] = []
        uploads: list[tuple[str, str]] = []  # (send-as name, local path)
        used_names: set[str] = set()
        for m in media_files:
            m = str(m)
            if m.startswith("sha256:"):
                refs.append(m)
                continue
            try:
                sha = blobs.sha256_file(m)
                if self.blob_exists(sha):
                    refs.append("sha256:" + sha)
                    continue
            except (OSError, RuntimeError):
                pass  # unreadable or probe failed: fall back to a plain upload
            name = Path(m).name
            while name in used_names:  # two distinct files sharing a basename
                name = "_" + name
            used_names.add(name)
            refs.append(name)
            uploads.append((name, m))
        if uploads:
            params["media"] = refs
            files = [("files", (name, open(path, "rb"))) for name, path in uploads]
            try:
                r = self._req("POST", "/v1/tasks", files=files,
                              data={"params": json.dumps(params)}, timeout=300)
            finally:
                for _, (_, fh) in files:
                    fh.close()
        else:
            # everything is already on the server: no multipart needed at all
            params["media_paths"] = refs
            r = self._req("POST", "/v1/tasks", json=params, timeout=60)
        return r.json()["id"]

    def task(self, tid: str) -> dict:
        return self._req("GET", f"/v1/tasks/{tid}").json()

    def tasks(self, status: str | None = None, model: str | None = None) -> list[dict]:
        q = []
        if status:
            q.append(f"status={status}")
        if model:
            q.append(f"model={model}")
        qs = ("?" + "&".join(q)) if q else ""
        return self._req("GET", f"/v1/tasks{qs}").json()

    def cancel(self, tid: str) -> dict:
        return self._req("DELETE", f"/v1/tasks/{tid}").json()

    def wait(self, tid: str, echo=None, poll_s: float = 2.0) -> dict:
        """Poll until terminal, streaming progress transitions via echo(line)."""
        last = ""
        while True:
            t = self.task(tid)
            p = t.get("progress") or {}
            line = f"[{t['status']}] {p.get('step', '')}: {p.get('detail', '')}"
            if p.get("chunk"):
                c = p["chunk"]
                line += f" (chunk {c['i']}/{c['n']})"
            if echo and line != last:
                echo(line)
                last = line
            if t["status"] in ("done", "failed", "canceled"):
                return t
            time.sleep(poll_s)
