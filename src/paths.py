# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""~/.aisee layout. Everything operational and temporary lives under one root."""

import os
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("AISEE_HOME", "~/.aisee")).expanduser()


def config_path() -> Path:
    return home() / "config.toml"


def creds_path() -> Path:
    return home() / "credentials.json"


def models_dir() -> Path:
    return home() / "models"


def hf_cache() -> Path:
    return home() / "hf-cache"


def jit_cache(image: str, image_id: str) -> Path:
    """Host-side kernel caches for one serving image build (see dockerctl._jit_cache_args).

    Model containers are recreated on every start, so without this each cold load
    re-JITs and re-autotunes its kernels."""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in image)
    return home() / "cache" / "jit" / f"{safe}-{image_id}"


def tasks_dir() -> Path:
    return home() / "tasks"


def tasks_db() -> Path:
    return tasks_dir() / "tasks.db"


def media_dir() -> Path:
    return tasks_dir() / "media"


def logs_dir() -> Path:
    return home() / "logs"


def model_logs_dir() -> Path:
    return logs_dir() / "models"


def api_log() -> Path:
    return logs_dir() / "api.log"


def run_dir() -> Path:
    return home() / "run"


def api_pidfile() -> Path:
    return run_dir() / "api.pid"


def ensure_layout() -> None:
    for d in (home(), models_dir(), hf_cache(), tasks_dir(), media_dir(),
              logs_dir(), model_logs_dir(), run_dir()):
        d.mkdir(parents=True, exist_ok=True)
