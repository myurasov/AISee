# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/describe — the API explains itself to an AI agent that has never seen it.

The markdown body is authored in res/describe.md (repo root); this module fills in the
dynamic parts: {{version}} and {{models}} (installed models merged with catalog
strengths/weaknesses/pitfalls and live state).
"""

from pathlib import Path

from . import __version__, catalog, registry

# res/describe.md lives next to the package dir (repo layout: src/ + res/); AISee is
# deployed from a source checkout, so this resolves both in dev and on hosts
_TEMPLATE_PATHS = [
    Path(__file__).resolve().parent.parent / "res" / "describe.md",
]

_FALLBACK = (
    "# AISee v{{version}}\n\nAISee is a tool that gives AI agents eyes. "
    "(res/describe.md template not found on this install - model guide below.)\n\n{{models}}\n"
)

_ENDPOINTS = [
    ("GET", "/v1/describe", "this document (markdown; ?format=json for structured)"),
    ("GET", "/v1/health", "liveness + per-model state summary"),
    ("GET", "/v1/models", "installed models: state, port, idle_timeout, default flag"),
    ("POST", "/v1/models/{slug}/start", "start a model (non-blocking; poll /v1/models)"),
    ("POST", "/v1/models/{slug}/stop", "stop a model (frees GPU memory; stays installed)"),
    ("POST", "/v1/tasks", "submit a query -> {id} (multipart: files + params JSON field; "
                          "or JSON with media_paths on the server host)"),
    ("GET", "/v1/tasks", "list tasks (?status=&model=)"),
    ("GET", "/v1/tasks/{id}", "full task: status, progress, timings, result"),
    ("DELETE", "/v1/tasks/{id}", "cancel a task"),
]


def _template() -> str:
    for p in _TEMPLATE_PATHS:
        if p.exists():
            return p.read_text()
    return _FALLBACK


def _model_lines(core) -> list[dict]:
    out = []
    for entry in registry.list_installed():
        cat = catalog.CATALOG.get(entry["slug"], {})
        v = core.model_view(entry)
        out.append({
            "slug": entry["slug"], "hf_id": entry["hf_id"], "state": v["state"],
            "default": v["default"],
            "supports_native_video": entry.get("supports_native_video", True),
            "strengths": cat.get("strengths", ""), "weaknesses": cat.get("weaknesses", ""),
            "pitfalls": cat.get("pitfalls", ""), "license": cat.get("license", ""),
        })
    return out


def _render_models(core) -> str:
    models = _model_lines(core)
    if not models:
        return "_No models installed yet (`aisee model install <slug>` on the host)._"
    lines: list[str] = []
    for m in models:
        flag = " **(default)**" if m["default"] else ""
        lines += [
            f"### `{m['slug']}`{flag} — {m['state']}",
            f"- HF id: `{m['hf_id']}`; native video: {'yes' if m['supports_native_video'] else 'no'}"
            + (f"; license: {m['license']}" if m["license"] else ""),
        ]
        if m["strengths"]:
            lines.append(f"- **Strengths:** {m['strengths']}")
        if m["weaknesses"]:
            lines.append(f"- **Weaknesses:** {m['weaknesses']}")
        if m["pitfalls"]:
            lines.append(f"- **Pitfalls:** {m['pitfalls']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def as_markdown(core) -> str:
    return (_template()
            .replace("{{version}}", __version__)
            .replace("{{models}}", _render_models(core)))


def as_json(core) -> dict:
    return {
        "name": "AISee", "version": __version__,
        "tagline": "AISee is a tool that gives AI agents eyes.",
        "endpoints": [{"method": m, "path": p, "purpose": d} for m, p, d in _ENDPOINTS],
        "task_kinds": ["look", "assert", "watch"],
        "statuses": ["queued", "preparing_media", "model_loading", "running",
                     "done", "failed", "canceled"],
        "models": _model_lines(core),
    }
