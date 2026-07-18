# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Global config: ~/.aisee/config.toml (read with tomllib, written with a tiny serializer)."""

import tomllib

from . import paths

# Sized for the main mode of operation: one resident model on a 96 GB-class GPU
# (or a GB10) with the dense serving profile (128k context, 16 images / 64 video frames).
DEFAULTS: dict = {
    "api": {"host": "0.0.0.0", "port": 8484},
    "defaults": {
        "default_model": "",
        "idle_timeout": 3600, # seconds; 0 = never unload
        "fps": 3.0,
        "frames": 16, # even-sampled frames per video (= the image budget)
        # answer budget knobs. 0 = unset: per-kind built-ins apply (assert 1024, watch
        # 4096/chunk, look 8192; reasoning models 8192 for every kind). A host may pin
        # max_tokens (all kinds) or max_tokens_look/assert/watch; per-call still wins.
        "max_tokens": 0,
        "request_timeout": 3600, # per-inference HTTP timeout (s); dense models with big answer budgets can run long
        "task_ttl_hours": 24, # finished tasks + their media are GC'd after this
        "blob_ttl_hours": 24, # content-addressed upload cache TTL; reuse refreshes it
    },
}


def lan_ip() -> str | None:
    """This host's outbound-interface IP (no packets sent)."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def _dump_toml(cfg: dict) -> str:
    lines: list[str] = []
    for section, values in cfg.items():
        lines.append(f"[{section}]")
        for k, v in values.items():
            if isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            else:
                lines.append(f'{k} = "{v}"')
        lines.append("")
    return "\n".join(lines)


def load() -> dict:
    cfg = {s: dict(v) for s, v in DEFAULTS.items()}
    p = paths.config_path()
    if p.exists():
        on_disk = tomllib.loads(p.read_text())
        for section, values in on_disk.items():
            cfg.setdefault(section, {}).update(values)
    return cfg


def save(cfg: dict) -> None:
    paths.ensure_layout()
    paths.config_path().write_text(_dump_toml(cfg))


def set_value(section: str, key: str, value) -> dict:
    cfg = load()
    cfg.setdefault(section, {})[key] = value
    save(cfg)
    return cfg
