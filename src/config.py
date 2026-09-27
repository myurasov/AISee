# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Global config: ~/.aisee/config.toml (read with tomllib, written with a tiny serializer)."""

import os
import threading
import tomllib

from . import paths

# bumped when load() must migrate older files (save() stamps it: every save writes it)
CONFIG_VERSION = 3

# Sized for the main mode of operation: one resident model on a 96 GB-class GPU
# (or a GB10) with the dense serving profile (128k context, 16 images / 96 video frames).
DEFAULTS: dict = {
    "meta": {"config_version": CONFIG_VERSION},
    "api": {"host": "0.0.0.0", "port": 4444},
    "defaults": {
        "default_model": "",
        "idle_timeout": 3600, # seconds; 0 = never unload
        # watch sampling rate: the video rate the Qwen3-VL/Cosmos family is trained at
        "fps": 2.0,
        "frames": 16, # even-sampled frames per video (= the image budget)
        # answer budget knobs. 0 = unset: per-kind built-ins apply (assert 1024, watch
        # 4096/chunk, look 8192; reasoning models 8192 for every kind). A host may pin
        # max_tokens (all kinds) or max_tokens_look/assert/watch; per-call still wins.
        "max_tokens": 0,
        "request_timeout": 3600, # per-inference HTTP timeout (s); dense models with big answer budgets can run long
        # discourages per-frame restatement loops in watch chunk answers (vLLM sampling
        # param; watch only - look/assert keep neutral sampling so OCR repetition like
        # table cells survives). Set to 1.0 (or 0) to disable.
        "watch_repetition_penalty": 1.1,
        # video-mode narration invents specific titles and share-state stories; risky
        # claims in chunk answers are cross-checked against a still frame (up to this
        # many checks per chunk; refuted claims are removed/replaced and the chunk is
        # flagged unstable). 0 disables.
        "watch_still_checks": 2,
        "task_ttl_hours": 24, # finished tasks + their media are GC'd after this
        "task_keep_max": 1000, # also keep at most this many finished tasks regardless of age; 0 = unlimited
        "blob_ttl_hours": 24, # content-addressed upload cache TTL; reuse refreshes it
        # per-capability audio defaults; empty = the single installed provider is used.
        # Set automatically when the first model with that capability is installed.
        "default_transcribe_model": "",
        "default_diarize_model": "",
        # chain-of-thought default for models with a thinking toggle (thinking_toggle=true
        # in the model TOML: the hybrid-template Qwen3.5-family entries): off, so a quick
        # check stays greedy and fast; per-call `thinking` wins. Always-on reasoning models
        # think regardless of this.
        "thinking": False,
    },
}


# the fps default before 1.1.0a3, which every save wrote verbatim into config.toml:
# version-1 files carrying it are migrated to the current default once. Watch used to
# deliver only 2 fps on the Qwen3-VL/Cosmos family whatever was set (their processor
# resampled each chunk); now that it delivers the configured rate, 3 fps chunks degraded
# that family's free-form narration in testing.
LEGACY_DEFAULT_FPS = 3.0


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
        try:
            version = int((on_disk.get("meta") or {}).get("config_version", 1))
        except (TypeError, ValueError):
            version = 1
        if version < CONFIG_VERSION:
            if version < 2 and cfg["defaults"].get("fps") == LEGACY_DEFAULT_FPS:
                cfg["defaults"]["fps"] = DEFAULTS["defaults"]["fps"]
            # before 1.1.0b1 no catalog model had a toggle, so a stored thinking = true is
            # the old default written out by a config save, not a choice
            if version < 3 and cfg["defaults"].get("thinking") is True:
                cfg["defaults"]["thinking"] = False
            cfg["meta"]["config_version"] = CONFIG_VERSION
            try:
                save(cfg)  # once, so the file shows what is in effect and a value set later sticks
            except OSError:
                pass  # read-only home: migrate in memory on every load
    return cfg


def save(cfg: dict) -> None:
    paths.ensure_layout()
    p = paths.config_path()
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(_dump_toml(cfg))
        os.replace(tmp, p)  # atomic: a concurrent load() never reads a half-written file
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def set_value(section: str, key: str, value) -> dict:
    cfg = load()
    cfg.setdefault(section, {})[key] = value
    save(cfg)
    return cfg
