# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Credentials: env var > CLI parameter > ~/.aisee/credentials.json > interactive prompt.

Values obtained from a CLI parameter or a prompt are persisted (600) so they are asked once.
Env vars win but are never persisted implicitly. The server never prompts.
"""

import getpass
import json
import os

from . import paths

KNOWN_KEYS = ("HF_TOKEN", "NGC_API_KEY", "AISEE_API_TOKEN")


def load_store() -> dict:
    p = paths.creds_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_store(store: dict) -> None:
    paths.ensure_layout()
    p = paths.creds_path()
    p.write_text(json.dumps(store, indent=2) + "\n")
    p.chmod(0o600)


def set_value(key: str, value: str) -> None:
    store = load_store()
    store[key] = value
    save_store(store)


def unset(key: str) -> bool:
    store = load_store()
    if key in store:
        del store[key]
        save_store(store)
        return True
    return False


def mask(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "…" + value[-4:]


def resolve(key: str, cli_value: str | None = None, interactive: bool = False) -> str | None:
    """Resolution order per spec §12. Persists CLI/prompted values."""
    v = os.environ.get(key)
    if v:
        return v
    if cli_value:
        set_value(key, cli_value)
        return cli_value
    v = load_store().get(key)
    if v:
        return v
    if interactive and os.isatty(0):
        v = getpass.getpass(f"{key} (hidden, stored in {paths.creds_path()}): ").strip()
        if v:
            set_value(key, v)
            return v
    return None
