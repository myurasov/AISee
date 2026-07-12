# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/describe - the API explains itself to an AI agent that has never seen it.

The markdown bodies are authored in res/describe.api.md (REST consumers) and
res/describe.mcp.md (MCP clients; tools only, no REST surface); this module fills in the
dynamic parts: {{version}} and {{models}} (installed models merged with catalog
strengths/weaknesses/pitfalls and live state).
"""

from pathlib import Path

from . import __version__, catalog, registry

# res/ lives next to the package dir (repo layout: src/ + res/); AISee is deployed from a
# source checkout, so this resolves both in dev and on hosts
_RES = Path(__file__).resolve().parent.parent / "res"
_TEMPLATES = {"api": _RES / "describe.api.md", "mcp": _RES / "describe.mcp.md"}

_FALLBACK = (
    "# AISee v{{version}}\n\nAISee is a tool that gives AI agents eyes. "
    "(res/describe.md template not found on this install - model guide below.)\n\n{{models}}\n"
)

# auth tiers: open (never needs a token), consumer (AISEE_API_TOKEN when set),
# admin (AISEE_ADMIN_TOKEN when set; consumer token gets 403)
_ENDPOINTS = [
    ("GET", "/v1/describe", "open", "this document (markdown; ?format=json for structured)"),
    ("GET", "/v1/health", "open", "liveness + per-model state summary"),
    ("GET", "/v1/gpu", "consumer", "live GPU stats: utilization, memory, power, temperature"),
    ("GET", "/v1/models", "consumer", "installed models: state, port, idle_timeout, default flag"),
    ("GET", "/v1/catalog", "consumer", "built-in model catalog with installed flags"),
    ("POST", "/v1/models", "admin", "install a model: {name: <catalog slug or HF id>, ...overrides}"),
    ("DELETE", "/v1/models/{slug}", "admin", "uninstall (weights stay cached)"),
    ("POST", "/v1/models/{slug}/start", "admin", "start a model (non-blocking; poll /v1/models; "
                                                 "409 if it would oversubscribe GPU memory)"),
    ("POST", "/v1/models/{slug}/stop", "admin", "stop a model (frees GPU memory; stays installed)"),
    ("POST", "/v1/tasks", "consumer", "submit a query -> {id} (multipart: files + params JSON "
                                      "field; or JSON with media_paths on the server host)"),
    ("GET", "/v1/tasks", "consumer", "list tasks (?status=&model=)"),
    ("GET", "/v1/tasks/{id}", "consumer", "full task: status, progress, timings "
                                          "(incl. total_s once finished), result"),
    ("DELETE", "/v1/tasks/{id}", "consumer", "cancel a task"),
    ("GET", "/v1/blobs/{sha256}", "consumer", "upload-dedup probe: {exists, size}"),
    ("POST", "/v1/blobs", "consumer", "upload media into the content-addressed store "
                                      "-> [{sha256, size}]; reference as sha256:<hash>"),
]


def _template(flavor: str) -> str:
    p = _TEMPLATES.get(flavor, _TEMPLATES["api"])
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
            "serving": {
                "max_model_len": entry.get("max_model_len"),
                "max_images": entry.get("max_images"),
                "video_frames": entry.get("video_frames"),
                "gpu_frac": entry.get("gpu_frac"),
                "concurrency": entry.get("concurrency", 1),
                "idle_timeout": entry.get("idle_timeout"),
            },
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
            f"### `{m['slug']}`{flag} - {m['state']}",
            f"- HF id: `{m['hf_id']}`; native video: {'yes' if m['supports_native_video'] else 'no'}"
            + (f"; license: {m['license']}" if m["license"] else ""),
            (lambda s: f"- Serving: context {s['max_model_len']} tokens; per request up to "
                       f"{s['max_images']} images / 1 video sampled to {s['video_frames']} frames; "
                       f"{s['concurrency']} concurrent inferences; gpu_frac {s['gpu_frac']}; "
                       f"idle unload after {s['idle_timeout']} s")(m["serving"]),
        ]
        if m["strengths"]:
            lines.append(f"- **Strengths:** {m['strengths']}")
        if m["weaknesses"]:
            lines.append(f"- **Weaknesses:** {m['weaknesses']}")
        if m["pitfalls"]:
            lines.append(f"- **Pitfalls:** {m['pitfalls']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def as_markdown(core, flavor: str = "api") -> str:
    return (_template(flavor)
            .replace("{{version}}", __version__)
            .replace("{{models}}", _render_models(core)))


def as_json(core) -> dict:
    return {
        "name": "AISee", "version": __version__,
        "tagline": "AISee is a tool that gives AI agents eyes.",
        "endpoints": [{"method": m, "path": p, "auth": a, "purpose": d}
                      for m, p, a, d in _ENDPOINTS],
        "auth": {"consumer": "AISEE_API_TOKEN (when set, guards query/read endpoints)",
                 "admin": "AISEE_ADMIN_TOKEN (when set, guards model management; "
                          "accepted everywhere)"},
        "task_kinds": ["look", "assert", "watch"],
        "statuses": ["queued", "preparing_media", "model_loading", "running",
                     "done", "failed", "canceled"],
        "models": _model_lines(core),
    }
