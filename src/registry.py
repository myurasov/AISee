# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Installed-model registry: one TOML file per model under ~/.aisee/models/<slug>.toml."""

import random
import socket
import subprocess
import tomllib

from . import catalog, config, paths


def _dump_entry(e: dict) -> str:
    lines = []
    for k, v in e.items():
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        elif isinstance(v, list):
            items = ", ".join('"' + str(i).replace('\\', '\\\\').replace('"', '\\"') + '"' for i in v)
            lines.append(f"{k} = [{items}]")
        else:
            s = str(v).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{k} = "{s}"')
    return "\n".join(lines) + "\n"


def gpu_profile() -> dict:
    """Detected GPU: name, memory, unified-ness, and the single-model gpu_frac default.

    On unified-memory systems (DGX Spark GB10, Grace-class) the GPU pool IS system RAM,
    so a slice is left for the OS and AISee itself; discrete VRAM is taken whole. When
    no GPU is detectable (client machine), a 96 GB discrete card is assumed.
    """
    name, mem_gib = "", 96.0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if out:
            name, mem = out.split("\n")[0].rsplit(",", 1)
            name = name.strip()
            try:
                mem_gib = float(mem) / 1024.0
            except ValueError:
                pass  # GB10 reports memory.total as [N/A]
    except (OSError, subprocess.TimeoutExpired):
        pass
    unified = any(k in name.upper() for k in ("GB10", "GH200", "GB200"))
    if unified and mem_gib == 96.0:
        mem_gib = 119.7  # GB10 hides memory.total; unified pool visible to CUDA
    frac = catalog.GPU_FRAC_UNIFIED if unified else catalog.GPU_FRAC_DISCRETE
    return {"name": name or "unknown", "mem_gib": mem_gib, "unified": unified,
            "gpu_frac": frac}


def default_gpu_frac() -> float:
    return gpu_profile()["gpu_frac"]


def fit_max_model_len(cat: dict, profile: dict, gpu_frac: float,
                      extra_args: list[str] | None = None) -> int:
    """Largest context whose KV cache fits next to the weights on this GPU.

    Capped at the checkpoint's native limit (ctx_native); an fp8 KV cache in the
    serve args halves the per-token KV cost, so the same GiB slice fits twice the
    context."""
    weights = cat.get("weights_gib")
    kv_128k = cat.get("kv_gib_128k")
    if not weights or not kv_128k:
        return catalog.DEFAULT_MAX_MODEL_LEN
    if extra_args and "fp8" in extra_args:
        kv_128k = kv_128k / 2.0
    budget = profile["mem_gib"] * gpu_frac - weights - catalog.ACTIVATION_HEADROOM_GIB
    native = int(cat.get("ctx_native", catalog.DEFAULT_MAX_MODEL_LEN))
    for cand in [native] + [c for c in catalog.CONTEXT_CANDIDATES if c < native]:
        if kv_128k * cand / 131072.0 <= budget:
            return cand
    return catalog.CONTEXT_CANDIDATES[-1]  # vLLM will complain at load if truly hopeless


def _free_port() -> int:
    for _ in range(80):
        port = random.randint(20000, 49151)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError("could not find a free port in 20000-49151")


def entry_path(slug: str):
    return paths.models_dir() / f"{slug}.toml"


def get(slug: str) -> dict | None:
    p = entry_path(slug)
    if not p.exists():
        return None
    return tomllib.loads(p.read_text())


def list_installed() -> list[dict]:
    out = []
    for p in sorted(paths.models_dir().glob("*.toml")):
        try:
            out.append(tomllib.loads(p.read_text()))
        except tomllib.TOMLDecodeError:
            continue
    return out


def install(name: str, *, image: str | None = None, gpu_frac: float | None = None,
            port: int | None = None, idle_timeout: int | None = None,
            extra_args: list[str] | None = None, max_model_len: int | None = None,
            concurrency: int | None = None) -> dict:
    """Resolve name against the catalog (or accept a raw HF id) and write the registry entry.

    Does not start the model. The port is chosen randomly once and persisted.
    """
    slug, cat = catalog.lookup(name)
    cat = cat or {}
    hf_id = cat.get("hf_id") or name
    if "/" not in hf_id:
        raise ValueError(f"'{name}' is not in the catalog; pass a full HF id like org/Model-Name")
    cfg = config.load()
    existing = get(slug) or {}
    profile = gpu_profile()
    engine = cat.get("engine", "vllm")
    if engine != "vllm":
        entry = {
            "slug": slug,
            "hf_id": hf_id,
            "image": image or cat["image"],
            "engine": engine,
            "modality": cat.get("modality", "audio"),
            "capabilities": list(cat.get("capabilities", [])),
            "port": port or existing.get("port") or _free_port(),
            "gpu_frac": gpu_frac if gpu_frac is not None else cat.get("gpu_frac", 0.06),
            "mem_gib": cat.get("mem_gib", 7),
            "mem_limit": cat.get("mem_limit", "16g"),
            "supports_native_video": False,
            "load_timeout": cat.get("load_timeout", 5400),
            "concurrency": concurrency or cat.get("concurrency", 1),
            "idle_timeout": idle_timeout if idle_timeout is not None
                            else existing.get("idle_timeout", cfg["defaults"]["idle_timeout"]),
        }
        paths.ensure_layout()
        entry_path(slug).write_text(_dump_entry(entry))
        _adopt_capability_defaults(entry, cfg)
        return entry
    # the catalog states an absolute requirement in GiB (portable across GPUs);
    # the serving fraction is derived from it for THIS host, clamped to the host cap.
    # An explicit --gpu-frac still overrides everything.
    mem_gib = cat.get("mem_gib") or catalog.mem_requirement_gib(cat)
    cap = (catalog.UNIFIED_CAPACITY_BUDGET if profile["unified"]
           else catalog.GPU_FRAC_DISCRETE)
    if gpu_frac is not None:
        frac = gpu_frac
    elif mem_gib:
        frac = round(min(cap, mem_gib / profile["mem_gib"]), 3)
    else:
        frac = cat.get("gpu_frac", profile["gpu_frac"])
    args = list(extra_args if extra_args is not None else cat.get("extra_args", []))
    # CUDA-graph capture is unstable/slow on unified-memory (GB10-class) systems, so they
    # serve eager; discrete GPUs keep graphs (measured 3-4x faster decode on RTX PRO 6000)
    if profile["unified"] and "--enforce-eager" not in args:
        args.append("--enforce-eager")
    ctx = (max_model_len or cat.get("max_model_len")
           or fit_max_model_len(cat, profile, frac, extra_args=args))
    entry = {
        "slug": slug,
        "hf_id": hf_id,
        "image": image or cat.get("image", catalog.DEFAULT_IMAGE),
        "port": port or existing.get("port") or _free_port(),
        "gpu_frac": frac,
        **({"mem_gib": mem_gib} if mem_gib else {}),
        "extra_args": args,
        "max_images": catalog.max_images_for(cat, ctx),
        "video_frames": cat.get("video_frames", catalog.DEFAULT_VIDEO_FRAMES),
        "max_model_len": ctx,
        "supports_native_video": cat.get("supports_native_video", True),
        "reasoning": cat.get("reasoning", False),
        "thinking_toggle": cat.get("thinking_toggle", False),
        "load_timeout": cat.get("load_timeout", 1800),
        "concurrency": concurrency or cat.get("concurrency", catalog.DEFAULT_CONCURRENCY),
        "idle_timeout": idle_timeout if idle_timeout is not None
                        else existing.get("idle_timeout", cfg["defaults"]["idle_timeout"]),
    }
    paths.ensure_layout()
    entry_path(slug).write_text(_dump_entry(entry))
    # first installed VISION model becomes the default (audio models never do)
    if not cfg["defaults"].get("default_model"):
        config.set_value("defaults", "default_model", slug)
    return entry


def _adopt_capability_defaults(entry: dict, cfg: dict) -> None:
    """First model installed for a capability becomes that capability's default."""
    for cap in entry.get("capabilities", []):
        key = f"default_{cap}_model"
        if not cfg["defaults"].get(key):
            config.set_value("defaults", key, entry["slug"])


def remove(slug: str) -> bool:
    p = entry_path(slug)
    if p.exists():
        entry = get(slug) or {}
        p.unlink()
        cfg = config.load()
        if cfg["defaults"].get("default_model") == slug:
            remaining = [e["slug"] for e in list_installed()
                         if e.get("modality", "vision") == "vision"]
            config.set_value("defaults", "default_model", remaining[0] if remaining else "")
        for cap in entry.get("capabilities", []):
            key = f"default_{cap}_model"
            if cfg["defaults"].get(key) == slug:
                remaining = [e["slug"] for e in list_installed()
                             if cap in (e.get("capabilities") or [])]
                config.set_value("defaults", key, remaining[0] if remaining else "")
        return True
    return False


def default_model() -> str | None:
    return config.load()["defaults"].get("default_model") or None


def default_for_capability(cap: str) -> str | None:
    """Configured default for a capability, else the single installed provider."""
    v = config.load()["defaults"].get(f"default_{cap}_model")
    if v and get(v):
        return v
    providers = [e["slug"] for e in list_installed() if cap in (e.get("capabilities") or [])]
    return providers[0] if providers else None
