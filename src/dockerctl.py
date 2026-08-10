# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Model containers: docker lifecycle, one container per model.

Engines: "vllm" (vision models, OpenAI-compatible) and the audio serving apps
("nemo-asr", "pyannote") built from res/serving/<dir>/ into local aisee/* images.
"""

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


# engine -> serving-app directory under res/serving/ (images built locally on the host)
ENGINE_BUILD_DIRS = {"nemo-asr": "audio-nemo", "pyannote": "audio-pyannote"}


def build_image(image: str, engine: str) -> None:
    """Build a local serving image from res/serving/<dir>/ (audio engines)."""
    from pathlib import Path
    ctx = Path(__file__).resolve().parent.parent / "res" / "serving" / ENGINE_BUILD_DIRS[engine]
    if not ctx.is_dir():
        raise RuntimeError(f"serving-app directory missing: {ctx}")
    _run(["build", "-t", image, str(ctx)], timeout=5400)


def health_url(entry: dict) -> str:
    """The readiness probe endpoint for this entry's engine."""
    path = "/v1/models" if entry.get("engine", "vllm") == "vllm" else "/health"
    return f"http://127.0.0.1:{entry['port']}{path}"


def start_model(entry: dict, hf_token: str | None = None) -> None:
    """(Re)create and start the container. Non-blocking: readiness is wait_ready()."""
    if entry.get("engine", "vllm") != "vllm":
        return _start_audio_model(entry, hf_token=hf_token)
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


def _start_audio_model(entry: dict, hf_token: str | None = None) -> None:
    """Audio serving container: small FastAPI app baked into a locally built image.

    GPU-gated at startup (the app exits nonzero when inference is not actually on
    CUDA), so wait_ready surfaces a CPU fallback loudly instead of serving it."""
    name = container_name(entry["slug"])
    port = int(entry["port"])
    mem = str(entry.get("mem_limit") or "16g")
    _run(["rm", "-f", name], check=False)
    args = [
        "run", "-d", "--name", name, "--restart", "unless-stopped",
        "--gpus", "all", "--ipc=host",
        # hard RAM cap: on unified-memory hosts a leaking/oversized job would otherwise
        # thrash the whole OS; hitting the cap kills the container visibly instead.
        # oom-score-adj makes audio containers the global OOM killer's first pick, so
        # host pressure never takes down sshd or the resident VLM
        "--memory", mem, "--memory-swap", mem, "--oom-score-adj", "500",
        "-e", "HF_HOME=/hf-cache",
        "-v", f"{paths.hf_cache()}:/hf-cache",
        "-p", f"{port}:{port}",
    ]
    if hf_token:
        args += ["-e", f"HF_TOKEN={hf_token}", "-e", f"HUGGING_FACE_HUB_TOKEN={hf_token}"]
    args += [entry["image"], "python", "/app/app.py",
             "--port", str(port), "--model", entry["hf_id"]]
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


def gpu_free_gib(unified: bool) -> float | None:
    """What is ACTUALLY free for a new model right now, in GiB.

    Unified memory (GB10 class): the GPU pool is system RAM -> MemAvailable.
    Discrete: total - used from nvidia-smi."""
    try:
        if unified:
            mi = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    mi[k] = v
            return float(mi["MemAvailable"].split()[0]) / 1048576.0
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        total, used = (float(x) for x in out.split(","))
        return (total - used) / 1024.0
    except Exception:
        return None  # no probe available: fall back to bookkeeping only


def restart_count(slug: str) -> int:
    r = _run(["inspect", "-f", "{{.RestartCount}}", container_name(slug)], check=False)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def wait_ready(entry: dict, timeout: int | None = None, progress=None) -> None:
    """Poll the engine's health endpoint until it serves; raise with a log tail on failure."""
    timeout = timeout or int(entry.get("load_timeout", 1800))
    url = health_url(entry)
    deadline = time.time() + timeout
    n = 0
    restarts0 = restart_count(entry["slug"])
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
        if restart_count(entry["slug"]) > restarts0 + 1:
            # restart policy is masking a crash loop (e.g. a failed GPU gate) - fail
            # fast with the reason instead of burning the whole load timeout
            raise RuntimeError(
                f"model container is crash-looping; last log lines:\n{logs_tail(entry['slug'])}")
        n += 1
        if progress and n % 4 == 0:
            progress(f"loading... {int(time.time() - (deadline - timeout))}s")
        time.sleep(5)
    raise RuntimeError(f"model not ready after {timeout}s (weights may still be downloading); "
                       f"log tail:\n{logs_tail(entry['slug'])}")
