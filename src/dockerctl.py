# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Model containers: docker lifecycle, one container per model.

Engines: "vllm" (vision models, OpenAI-compatible) and the audio serving apps
("nemo-asr", "pyannote") built from res/serving/<dir>/ into local aisee/* images.
"""

import base64
import json
import re
import shutil
import subprocess
import time

import httpx

from . import catalog, paths

# vLLM 26.06 image bug: prometheus-fastapi-instrumentator 8.0.0 crashes on routers without
# .path, 500-ing every request. Patched None-safe inside the container after start; a no-op
# on images that already ship the fix (26.08+), kept for per-model --image overrides.
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


def container_image(slug: str) -> str | None:
    """Image the model's container was created from (None when there is no container)."""
    try:
        r = _run(["inspect", "-f", "{{.Config.Image}}", container_name(slug)], check=False)
    except FileNotFoundError:
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


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


def _jit_cache_args(image: str) -> list[str]:
    """docker-run args that keep kernel caches across container recreation.

    Persists Triton kernels and vLLM's FlashInfer autotune results (files their runtimes
    write atomically) and the CUDA driver's PTX JIT cache; vLLM's torch.compile cache stays
    inside the container. Measured on a GB10 with Nemotron NVFP4: engine init 207 -> 44 s,
    and its first request after a load ~140 s -> seconds (vLLM's FlashAttention ships no
    SASS for the GB10's sm_121, so the driver compiles its PTX on first use). Discrete
    Blackwell cards get native SASS (the CUDA cache stays empty) but still reuse Triton
    kernels: Cosmos3-Nano's first request on an RTX PRO 6000 ~10 s -> ~1.5 s. NGC vLLM and
    vllm-omni images only (they run as root and bake nothing at these paths). Keyed by
    image id, so a re-pushed tag never reuses kernels built against another stack. Best
    effort: any failure just means a re-JIT. Tuning picks persist until the directory is
    removed (`sudo rm -rf ~/.aisee/cache/jit`); directories of image builds no longer on the
    host are pruned when the API starts (prune_jit_caches)."""
    if not image.startswith(("nvcr.io/nvidia/vllm", "vllm/vllm-omni")):
        return []
    try:
        r = _run(["image", "inspect", "-f", "{{.Id}}", image], check=False)
    except FileNotFoundError:
        return []
    image_id = r.stdout.strip().rpartition(":")[2][:12] if r.returncode == 0 else ""
    if not image_id:
        return []
    jit = paths.jit_cache(image, image_id)
    try:
        jit.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []
    return ["-v", f"{jit}:/aisee-jit",
            "-e", "TRITON_CACHE_DIR=/aisee-jit/triton",
            "-e", "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/aisee-jit/flashinfer-autotune",
            "-e", "CUDA_CACHE_PATH=/aisee-jit/cuda",
            # one model writes ~55-125 MB (Nemotron, Cosmos3-Nano); the cap bounds what
            # all models on one image build accumulate in their shared directory
            "-e", f"CUDA_CACHE_MAXSIZE={1 << 30}"]


def prune_jit_caches() -> tuple[list[str], list[str]]:
    """Remove the kernel-cache directories of serving image builds no longer on this host.

    Each directory is keyed by image id (paths.jit_cache), so once a build leaves the host
    (`docker rmi`, or `docker image prune` after a re-pull left it untagged) nothing reads its
    directory again. The model containers write these files as root, so what this user cannot
    delete goes through a throwaway container of the default serving image (never pulled for
    this). Best effort, and conservative: no trustworthy view of the images or containers means
    no pruning, and a directory any container mounts is kept. Returns (removed, left) names -
    left = root-owned directories that could not be removed."""
    root = paths.home() / "cache" / "jit"
    if not root.is_dir():
        return [], []
    # snapshot before listing images: a directory created after this (a model starting on a
    # freshly pulled image) is never a candidate
    dirs = [d for d in root.iterdir() if d.is_dir() and not d.is_symlink()
            and re.fullmatch(r".+-[0-9a-f]{12}", d.name)]
    if not dirs:
        return [], []
    try:
        imgs = _run(["images", "-a", "--no-trunc", "--format", "{{.ID}}"], check=False, timeout=60)
        ps = _run(["ps", "-aq"], check=False, timeout=60)
        if imgs.returncode or ps.returncode:
            return [], []
        mounted: set[str] = set()
        if ps.stdout.split():
            ins = _run(["inspect", "-f", "{{range .Mounts}}{{.Source}}\n{{end}}", *ps.stdout.split()],
                       check=False, timeout=60)
            if ins.returncode:
                return [], []  # a container vanished mid-scan: try again next start
            mounted = {ln.strip() for ln in ins.stdout.splitlines() if ln.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return [], []
    live = {m.group(1)[:12] for m in re.finditer(r"sha256:([0-9a-f]{64})", imgs.stdout)}
    if not live:
        return [], []  # never prune blind
    stale = [d for d in dirs if d.name[-12:] not in live and str(d) not in mounted]
    removed, left = [], []
    for d in stale:
        shutil.rmtree(d, ignore_errors=True)
        (left if d.exists() else removed).append(d.name)
    if left:
        try:
            ok = _run(["image", "inspect", catalog.DEFAULT_IMAGE], check=False,
                      timeout=60).returncode == 0
            if ok and _run(["run", "--rm", "--pull", "never", "--network", "none",
                            "--entrypoint", "rm", "-v", f"{root}:/jit", catalog.DEFAULT_IMAGE,
                            "-rf", *[f"/jit/{n}" for n in left]],
                           check=False, timeout=300).returncode == 0:
                removed, left = removed + left, []
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return removed, left


# eager safetensors loading (see start_model): measured on a GB10 with the NGC and vllm-omni
# images; it holds one whole shard in RAM (~2x while unpacking), outside the capacity check,
# so it is used only for cached checkpoints whose largest shard stays small (catalog: <= 5 GiB)
EAGER_LOAD_MAX_SHARD_GIB = 6
_EAGER_IMAGES = ("nvcr.io/nvidia/vllm:", "vllm/vllm-omni:")


def _largest_shard_gib(hf_id: str) -> float | None:
    """Largest cached .safetensors file of a checkpoint (GiB), None when not downloaded."""
    snaps = paths.hf_cache() / "hub" / f"models--{hf_id.replace('/', '--')}" / "snapshots"
    try:
        sizes = [f.stat().st_size for f in snaps.rglob("*.safetensors")]
    except OSError:
        return None
    return max(sizes) / (1 << 30) if sizes else None


def _eager_load(entry: dict) -> bool:
    """Eager weight loading for this start: a GB10, a tested image family, a cached checkpoint
    with small shards (the first start downloads and loads the default way)."""
    if not str(entry.get("image", "")).startswith(_EAGER_IMAGES):
        return False
    try:
        from . import registry  # lazy: registry pulls in more than start_model needs
        if "GB10" not in registry.gpu_profile()["name"].upper():
            return False
    except Exception:  # noqa: BLE001 - no GPU view: keep vLLM's default loading
        return False
    largest = _largest_shard_gib(entry["hf_id"])
    return largest is not None and largest <= EAGER_LOAD_MAX_SHARD_GIB


def start_model(entry: dict, hf_token: str | None = None) -> None:
    """(Re)create and start the container. Non-blocking: readiness is wait_ready()."""
    if entry.get("engine", "vllm") != "vllm":
        return _start_audio_model(entry, hf_token=hf_token)
    name = container_name(entry["slug"])
    port = int(entry["port"])
    video_io = {"num_frames": entry["video_frames"]}
    # see catalog.UNIFORM_VIDEO_LOADER
    loader = catalog.video_loader(entry)
    if loader:
        video_io["video_backend"] = loader
    serve = [
        "vllm", "serve", entry["hf_id"],
        "--host", "0.0.0.0", "--port", str(port),
        "--gpu-memory-utilization", str(entry["gpu_frac"]),
        "--max-model-len", str(entry["max_model_len"]),
        "--limit-mm-per-prompt", json.dumps({"image": entry["max_images"], "video": 1}),
        "--media-io-kwargs", json.dumps({"video": video_io}),
        # the mm processor cache desyncs between vLLM's frontend and engine when a client
        # disconnect aborts an in-flight request, then 500s forever on that media hash
        # ("Expected a cached item for mm_hash=..."); re-preprocessing is cheap - disable it
        "--mm-processor-cache-gb", "0",
    ] + list(entry.get("extra_args", []))
    # vLLM reads --max_num_seqs as --max-num-seqs; a --config YAML may set it too
    flags = {str(a).split("=")[0].replace("_", "-") for a in serve if str(a).startswith("--")}
    if not flags & {"--max-num-seqs", "--config"}:
        # AISee keeps at most concurrency^2 requests in flight per model (task workers x
        # watch chunk threads). vLLM's default (1024 on >= 70 GB GPUs) sizes CUDA graphs
        # and vLLM 0.27's post-profile sampler warmup (~0.6 GB, outside the memory budget)
        # for that many - enough to OOM Qwen3-VL-32B on a 96 GB card at gpu_frac 0.97.
        # Fixed at container creation: a concurrency change needs a model stop to apply
        conc = max(1, int(entry.get("concurrency", 1)))
        serve += ["--max-num-seqs", str(min(max(16, conc * conc), 256))]
    if not flags & {"--safetensors-load-strategy", "--load-format", "--model-loader-extra-config",
                    "--config"} and _eager_load(entry):
        # GB10 unified memory: vLLM's default mmap load is page-fault bound there - a 51 GiB
        # checkpoint took 322-345 s cold and 324 s warm, reading each shard into memory first
        # ("eager") 61 s cold / 40 s warm (Qwen3.8-27B, 2026-09-28). Other GPUs keep vLLM's
        # default; a strategy (or loader) set in extra_args wins
        serve += ["--safetensors-load-strategy", "eager"]
    _run(["rm", "-f", name], check=False)
    args = [
        "run", "-d", "--name", name, "--restart", "unless-stopped",
        "--gpus", "all", "--ipc=host", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-e", "HF_HOME=/hf-cache",
        "-v", f"{paths.hf_cache()}:/hf-cache",
        "-p", f"{port}:{port}",
    ]
    args += _jit_cache_args(entry["image"])
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
    """Apply the instrumentator patch (nvcr vLLM images) and restart so it takes effect.

    Restarts only when the patch actually changed a file: images that already ship the
    fix (26.08+) would otherwise pay a pointless container restart on every start."""
    if not entry["image"].startswith("nvcr.io/nvidia/vllm"):
        return False
    name = container_name(entry["slug"])
    b64 = base64.b64encode(_INSTRUMENTATOR_PATCH.encode()).decode()
    deadline = time.time() + wait_s
    while time.time() < deadline:
        r = _run(["exec", name, "python3", "-c",
                  f"import base64; exec(base64.b64decode('{b64}').decode())"], check=False)
        if r.returncode == 0:
            if "patched" not in r.stdout.split():
                return False
            _run(["restart", name])
            return True
        if container_state(entry["slug"]) == "absent":
            return False  # removed (stopped) meanwhile: nothing left to patch
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
            progress(f"starting... {int(time.time() - (deadline - timeout))}s")
        time.sleep(5)
    raise RuntimeError(f"model not ready after {timeout}s (weights may still be downloading); "
                       f"log tail:\n{logs_tail(entry['slug'])}")
