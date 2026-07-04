# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Installed-model registry: one TOML file per model under ~/.aisee/models/<slug>.toml."""

import random
import socket
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
            extra_args: list[str] | None = None) -> dict:
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
    entry = {
        "slug": slug,
        "hf_id": hf_id,
        "image": image or cat.get("image", catalog.DEFAULT_IMAGE),
        "port": port or existing.get("port") or _free_port(),
        "gpu_frac": gpu_frac if gpu_frac is not None else cat.get("gpu_frac", 0.85),
        "extra_args": extra_args if extra_args is not None else cat.get("extra_args", []),
        "max_images": cat.get("max_images", 8),
        "video_frames": cat.get("video_frames", 16),
        "max_model_len": cat.get("max_model_len", 32768),
        "supports_native_video": cat.get("supports_native_video", True),
        "reasoning": cat.get("reasoning", False),
        "load_timeout": cat.get("load_timeout", 1800),
        "idle_timeout": idle_timeout if idle_timeout is not None
                        else existing.get("idle_timeout", cfg["defaults"]["idle_timeout"]),
    }
    paths.ensure_layout()
    entry_path(slug).write_text(_dump_entry(entry))
    # first installed model becomes the default
    if not cfg["defaults"].get("default_model"):
        config.set_value("defaults", "default_model", slug)
    return entry


def remove(slug: str) -> bool:
    p = entry_path(slug)
    if p.exists():
        p.unlink()
        cfg = config.load()
        if cfg["defaults"].get("default_model") == slug:
            remaining = [e["slug"] for e in list_installed()]
            config.set_value("defaults", "default_model", remaining[0] if remaining else "")
        return True
    return False


def default_model() -> str | None:
    return config.load()["defaults"].get("default_model") or None
