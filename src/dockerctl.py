# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Model containers: docker lifecycle for vLLM serving instances (one container per model)."""

import base64
import json
import subprocess
import time

import httpx

from . import paths

# vLLM 26.06 image bug: prometheus-fastapi-instrumentator 8.0.0 crashes on routers without
# .path, 500-ing every request. Patched None-safe inside the container after start.
_INSTRUMENTATOR_PATCH = """
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/prometheus_fastapi_instrumentator/routing.py")
if p.exists():
    s = p.read_text()
    s2 = s.replace("route_name = route.path", 'route_name = getattr(route, "path", None)')
    s2 = s2.replace("route_name += child_route_name", 'route_name = (route_name or "") + child_route_name')
    if s2 != s:
        p.write_text(s2)
        print("patched")
"""


def container_name(slug: str) -> str:
    return f"aisee-{slug}"


def _run(args: list[str], check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess:
    r = subprocess.run(["docker"] + args, check=False, capture_output=True, text=True,
                       timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {(r.stderr or r.stdout).strip()[-500:]}")
    return r


def docker_available() -> bool:
    try:
        _run(["info", "--format", "{{.ServerVersion}}"], timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def container_state(slug: str) -> str:
    """'running' | 'exited' | 'absent'"""
    try:
        r = _run(["inspect", "-f", "{{.State.Running}}", container_name(slug)], check=False)
    except FileNotFoundError:
        return "absent"
    if r.returncode != 0:
        return "absent"
    return "running" if r.stdout.strip() == "true" else "exited"


def list_aisee_containers() -> list[str]:
    r = _run(["ps", "-a", "--filter", "name=aisee-", "--format", "{{.Names}}"], check=False)
    return [n for n in r.stdout.split() if n.startswith("aisee-")]


def logs_tail(slug: str, n: int = 40) -> str:
    r = _run(["logs", "--tail", str(n), container_name(slug)], check=False)
    return (r.stdout + r.stderr)[-8000:]


def login_nvcr(ngc_key: str) -> None:
    subprocess.run(["docker", "login", "nvcr.io", "-u", "$oauthtoken", "--password-stdin"],
                   input=ngc_key, text=True, check=True, capture_output=True)


def pull(image: str, ngc_key: str | None = None) -> None:
    if image.startswith("nvcr.io/") and ngc_key:
        login_nvcr(ngc_key)
    _run(["pull", image], timeout=3600)


def image_present(image: str) -> bool:
    r = _run(["images", "-q", image], check=False)
    return bool(r.stdout.strip())


def start_model(entry: dict, hf_token: str | None = None) -> None:
    """(Re)create and start the container. Non-blocking: readiness is wait_ready()."""
    name = container_name(entry["slug"])
    port = int(entry["port"])
    serve = [
        "vllm", "serve", entry["hf_id"],
        "--host", "0.0.0.0", "--port", str(port),
        "--gpu-memory-utilization", str(entry["gpu_frac"]),
        "--max-model-len", str(entry["max_model_len"]),
        "--limit-mm-per-prompt", json.dumps({"image": entry["max_images"], "video": 1}),
        "--media-io-kwargs", json.dumps({"video": {"num_frames": entry["video_frames"]}}),
        # the mm processor cache desyncs between vLLM's frontend and engine when a client
        # disconnect aborts an in-flight request, then 500s forever on that media hash
        # ("Expected a cached item for mm_hash=..."); re-preprocessing is cheap - disable it
        "--mm-processor-cache-gb", "0",
    ] + list(entry.get("extra_args", []))
    _run(["rm", "-f", name], check=False)
    args = [
        "run", "-d", "--name", name, "--restart", "unless-stopped",
        "--gpus", "all", "--ipc=host", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-e", "HF_HOME=/hf-cache",
        "-v", f"{paths.hf_cache()}:/hf-cache",
        "-p", f"{port}:{port}",
    ]
    if hf_token:
        args += ["-e", f"HF_TOKEN={hf_token}", "-e", f"HUGGING_FACE_HUB_TOKEN={hf_token}"]
    args += [entry["image"]] + serve
    _run(args)


def apply_image_patches(entry: dict, wait_s: int = 150) -> bool:
    """Apply the instrumentator patch (nvcr vLLM images) and restart so it takes effect."""
    if not entry["image"].startswith("nvcr.io/nvidia/vllm"):
        return False
    name = container_name(entry["slug"])
    b64 = base64.b64encode(_INSTRUMENTATOR_PATCH.encode()).decode()
    deadline = time.time() + wait_s
    while time.time() < deadline:
        r = _run(["exec", name, "python3", "-c",
                  f"import base64; exec(base64.b64decode('{b64}').decode())"], check=False)
        if r.returncode == 0:
            _run(["restart", name])
            return True
        time.sleep(5)
    return False


def stop_model(slug: str) -> None:
    """Stop the container (GPU memory freed); weights and registry entry kept."""
    _run(["rm", "-f", container_name(slug)], check=False)


def wait_ready(entry: dict, timeout: int | None = None, progress=None) -> None:
    """Poll the vLLM endpoint until it serves /v1/models; raise with a log tail on failure."""
    timeout = timeout or int(entry.get("load_timeout", 1800))
    url = f"http://127.0.0.1:{entry['port']}/v1/models"
    deadline = time.time() + timeout
    n = 0
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=5)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if container_state(entry["slug"]) != "running":
            raise RuntimeError(
                f"model container exited during load; last log lines:\n{logs_tail(entry['slug'])}")
        n += 1
        if progress and n % 4 == 0:
            progress(f"model is loading ({int(time.time() - (deadline - timeout))}s elapsed)")
        time.sleep(5)
    raise RuntimeError(f"model not ready after {timeout}s (weights may still be downloading); "
                       f"log tail:\n{logs_tail(entry['slug'])}")
