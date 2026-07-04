# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Global config: ~/.aisee/config.toml (read with tomllib, written with a tiny serializer)."""

import tomllib

from . import paths

DEFAULTS: dict = {
    "api": {"host": "0.0.0.0", "port": 8484},
    "defaults": {
        "default_model": "",
        "idle_timeout": 900,          # seconds; 0 = never unload
        "fps": 1.0,
        "frames": 8,
        "max_tokens": 1024,
        "request_timeout": 600,       # per-inference HTTP timeout (s)
        "task_retention_days": 7,
    },
}


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
