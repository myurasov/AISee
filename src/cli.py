# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""The `aisee` CLI — single entry point. Everything except the bootstrap set
(install / uninstall / api ...) is a thin wrapper over the REST API."""

import argparse
import json
import shutil
import subprocess
import sys

from . import __version__, config, creds, dockerctl, paths, registry
from .client import Client, is_local


def _p(msg: str) -> None:
    print(msg, flush=True)


def _lan_ip() -> str | None:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # no packets sent; just picks the outbound interface
            return s.getsockname()[0]
    except OSError:
        return None


def _listen_line() -> str:
    cfg = config.load()["api"]
    line = f"listening on {cfg['host']}:{cfg['port']}"
    if cfg["host"] == "0.0.0.0":
        ip = _lan_ip()
        line += f" — reachable at http://{ip or '<this-host-ip>'}:{cfg['port']}"
    return line


def _client(args) -> Client:
    c = Client(server=getattr(args, "server", None),
               autostart=not getattr(args, "no_autostart", False),
               token=getattr(args, "token", None))
    c.ensure()
    return c


# ---------------- bootstrap commands (necessarily local) ----------------

def cmd_install(args) -> int:
    ok = True
    for tool, hint in (("docker", "install docker + NVIDIA container toolkit"),
                       ("ffmpeg", "apt install ffmpeg"), ("ffprobe", "apt install ffmpeg")):
        if shutil.which(tool):
            _p(f"  ok: {tool}")
        else:
            _p(f"  MISSING: {tool} — {hint}")
            ok = False
    if shutil.which("docker") and not dockerctl.docker_available():
        _p("  MISSING: docker daemon not reachable (is it running? are you in the docker group?)")
        ok = False
    if shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True)
        _p(f"  gpu: {r.stdout.strip() or 'nvidia-smi present but no GPU listed'}")
        # docker must be able to hand the GPU to containers (NVIDIA Container Toolkit):
        # docker 25+ resolves --gpus via CDI or the nvidia runtime
        info = subprocess.run(["docker", "info"], capture_output=True, text=True).stdout
        import pathlib as _pl
        cdi_ok = any(_pl.Path(d, "nvidia.yaml").exists() for d in ("/etc/cdi", "/var/run/cdi"))
        if "nvidia" not in info and not cdi_ok:
            _p("  MISSING: NVIDIA Container Toolkit (docker cannot pass the GPU to containers).")
            _p("           Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html")
            _p("           then: sudo nvidia-ctk runtime configure --runtime=docker &&")
            _p("                 sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml &&")
            _p("                 sudo systemctl restart docker")
            ok = False
    else:
        _p("  warning: nvidia-smi not found — model containers will not get a GPU")
    paths.ensure_layout()
    if not paths.config_path().exists():
        config.save(config.load())
    _p(f"  initialized {paths.home()}")
    for m in args.with_models or []:
        e = registry.install(m)
        _p(f"  model installed (registry): {e['slug']} (port {e['port']})")
    if not ok:
        _p("install: components missing (see above)")
        return 1
    _p("install: ok")
    return 0


def cmd_uninstall(args) -> int:
    target = paths.home()
    if not args.yes:
        resp = input(f"Remove ALL AISee state ({target}, containers aisee-*)"
                     f"{' except hf-cache' if args.keep_cache else ''}? [y/N] ")
        if resp.strip().lower() != "y":
            _p("aborted")
            return 1
    Client(autostart=False).api_stop()
    for name in dockerctl.list_aisee_containers():
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        _p(f"  removed container {name}")
    if args.keep_cache:
        for child in target.iterdir():
            if child.name != "hf-cache":
                shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink()
        _p(f"  removed {target} (kept hf-cache/)")
    else:
        shutil.rmtree(target, ignore_errors=True)
        _p(f"  removed {target}")
    return 0


def cmd_api(args) -> int:
    c = Client(autostart=False)
    if args.api_cmd == "start":
        if args.port:
            config.set_value("api", "port", int(args.port))
        if args.host:
            config.set_value("api", "host", args.host)
        if args.port:
            c = Client(autostart=False)
        if c.api_running():
            _p(f"already running at {c.base}")
            return 0
        c.api_start()
        _p(f"API started: {_listen_line()} (log: {paths.api_log()})")
        return 0
    if args.api_cmd == "stop":
        _p("stopped" if c.api_stop() else "not running (no pidfile)")
        return 0
    if c.api_running():
        h = c.health()
        _p(f"running at {c.base} (v{h['version']}); models: {json.dumps(h['models'])}")
        return 0
    cfg = config.load()["api"]
    _p(f"not running; `aisee api start` will listen on {cfg['host']}:{cfg['port']}")
    return 1


# ---------------- API-backed commands ----------------

def cmd_status(args) -> int:
    c = Client(server=getattr(args, "server", None), autostart=False,
               token=getattr(args, "token", None))
    if not c.api_running():
        if is_local(c.base):
            cfg = config.load()["api"]
            _p(f"API: down - local daemon not running; `aisee api start` will listen on "
               f"{cfg['host']}:{cfg['port']}")
        else:
            _p(f"API: down at {c.base}")
        return 1
    h = c.health()
    where = _listen_line() if is_local(c.base) else f"at {c.base}"
    _p(f"API: up, {where} (v{h['version']})")
    for m in c.models():
        d = " [default]" if m["default"] else ""
        _p(f"  {m['slug']}{d}: {m['state']} (port {m['port']}, idle_timeout {m['idle_timeout']}s)")
    open_tasks = [t for t in c.tasks() if t["status"] not in ("done", "failed", "canceled")]
    _p(f"  open tasks: {len(open_tasks)}")
    try:
        du = subprocess.run(["du", "-sh", str(paths.hf_cache())], capture_output=True, text=True)
        _p(f"  hf-cache: {du.stdout.split()[0] if du.stdout else '?'}")
    except OSError:
        pass
    return 0


def cmd_model(args) -> int:
    if args.model_cmd == "install":
        entry = registry.install(args.name, image=args.image, gpu_frac=args.gpu_frac,
                                 port=args.port, idle_timeout=args.idle_timeout,
                                 extra_args=args.arg if args.arg else None,
                                 max_model_len=args.max_model_len,
                                 concurrency=args.concurrency)
        if args.hf_token:
            creds.set_value("HF_TOKEN", args.hf_token)
        prof = registry.gpu_profile()
        _p(f"installed (registry): {entry['slug']} -> {entry['hf_id']}")
        from . import catalog as _cat
        w = _cat.CATALOG.get(entry["slug"], {}).get("weights_gib")
        if w and w + _cat.ACTIVATION_HEADROOM_GIB > prof["mem_gib"] * entry["gpu_frac"]:
            _p(f"  WARNING: ~{w} GiB of weights will not fit this GPU "
               f"({prof['mem_gib']:.0f} GiB) - the model will fail to load")
        _p(f"  gpu: {prof['name']} ({prof['mem_gib']:.0f} GiB{', unified' if prof['unified'] else ''}) "
           f"-> gpu_frac {entry['gpu_frac']}, context {entry['max_model_len']} tokens")
        _p(f"  image {entry['image']}, port {entry['port']}")
        _p("start it with: aisee model start " + entry["slug"])
        return 0
    if args.model_cmd == "remove":
        dockerctl.stop_model(args.slug)
        ok = registry.remove(args.slug)
        _p(f"removed {args.slug}" if ok else f"{args.slug} was not installed")
        return 0 if ok else 1
    if args.model_cmd == "list":
        c = Client(server=getattr(args, "server", None), autostart=False,
                   token=getattr(args, "token", None))
        if c.api_running():
            for m in c.models():
                d = " [default]" if m["default"] else ""
                note = f" ({m['loading_note']})" if m.get("loading_note") else ""
                _p(f"{m['slug']}{d}: {m['state']}{note} port={m['port']}")
        else:
            state_names = {"absent": "installed", "exited": "failed", "running": "running"}
            for e in registry.list_installed():
                d = " [default]" if registry.default_model() == e["slug"] else ""
                st = state_names[dockerctl.container_state(e["slug"])]
                _p(f"{e['slug']}{d}: {st} port={e['port']}")
        return 0
    if args.model_cmd == "start":
        c = _client(args)
        r = c.model_start(args.slug)
        _p(f"{args.slug}: {r['state']} (poll `aisee model list`; cold load can take minutes)")
        return 0
    if args.model_cmd == "stop":
        c = _client(args)
        r = c.model_stop(args.slug)
        _p(f"{args.slug}: {r['state']}")
        return 0
    if args.model_cmd == "logs":
        print(dockerctl.logs_tail(args.slug, args.n))
        return 0
    if args.model_cmd == "default":
        if not registry.get(args.slug):
            _p(f"{args.slug} is not installed")
            return 1
        config.set_value("defaults", "default_model", args.slug)
        _p(f"default model: {args.slug}")
        return 0
    return 1


def cmd_creds(args) -> int:
    if args.creds_cmd == "set":
        value = args.value
        if not value:
            import getpass
            value = getpass.getpass(f"{args.key} (hidden): ").strip()
        if not value:
            _p("no value given")
            return 1
        creds.set_value(args.key, value)
        _p(f"{args.key} stored in {paths.creds_path()}")
        return 0
    if args.creds_cmd == "unset":
        _p("removed" if creds.unset(args.key) else "not set")
        return 0
    store = creds.load_store()
    if not store:
        _p("(no stored credentials)")
    for k, v in store.items():
        _p(f"{k} = {creds.mask(v)}")
    return 0


def _query_params(args, kind: str) -> dict:
    params: dict = {}
    if kind == "look" or (kind == "watch" and args.question):
        params["question"] = args.question
    if kind == "assert" or (kind == "watch" and getattr(args, "expectation", None)):
        params["expectation"] = args.expectation
    for k in ("model", "frames", "fps", "max_tokens", "chunk_seconds", "scale", "server_frames"):
        v = getattr(args, k.replace("-", "_"), None)
        if v is not None:
            params[k] = v
    if getattr(args, "native", False):
        params["native"] = True
    ctx = getattr(args, "context", None)
    if getattr(args, "context_file", None):
        ctx = ((ctx + "\n") if ctx else "") + open(args.context_file).read()
    if ctx:
        params["context"] = ctx
    return params


def cmd_query(args, kind: str) -> int:
    c = _client(args)
    params = _query_params(args, kind)
    tid = c.submit(kind, args.media, params)
    if args.no_wait:
        _p(tid)
        return 0
    _p(f"task {tid}")
    t = c.wait(tid, echo=lambda line: _p(f"  {line}"))
    if t["status"] == "failed":
        _p(f"FAILED: {t['error']['message'] if t.get('error') else 'unknown error'}")
        return 2
    if t["status"] == "canceled":
        _p("canceled")
        return 3
    r = t["result"]
    timings = t.get("timings", {})
    if kind == "look":
        _p(r["answer"])
    else:
        _p(json.dumps(r, indent=2))
    inf = timings.get("inference_s")
    load = timings.get("model_load_s")
    note = " ".join(filter(None, [f"inference {inf}s" if inf else "",
                                  f"model load {load}s" if load else ""]))
    if note:
        _p(f"  ({note})")
    if kind == "assert" or (kind == "watch" and "pass" in r):
        return 0 if r.get("pass") else 1
    return 0


def cmd_task(args) -> int:
    c = _client(args)
    if args.task_cmd == "list":
        for t in c.tasks(status=args.status, model=args.model):
            _p(f"{t['id']} {t['kind']:6s} {t['model']:30s} {t['status']:16s} "
               f"{(t.get('progress') or {}).get('detail', '')[:50]}")
        return 0
    if args.task_cmd == "show":
        print(json.dumps(c.task(args.id), indent=2))
        return 0
    if args.task_cmd == "cancel":
        _p(json.dumps(c.cancel(args.id)))
        return 0
    return 1


def cmd_describe(args) -> int:
    c = _client(args)
    print(c.describe("json" if args.json else "markdown"))
    return 0


def cmd_mcp(args) -> int:
    from . import mcp_server
    mcp_server.main(server=args.server)
    return 0


# ---------------- parser ----------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aisee",
                                 description="AISee is a tool that gives AI agents eyes.")
    ap.add_argument("--version", action="version", version=f"aisee {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_server(p):
        p.add_argument("--server", help="API base URL (default: local; or AISEE_SERVER env)")
        p.add_argument("--token", help="bearer token (default: AISEE_ADMIN_TOKEN then "
                                       "AISEE_API_TOKEN, from env or the creds store)")
        p.add_argument("--no-autostart", action="store_true",
                       help="do not auto-start a local API daemon")

    p = sub.add_parser("install", help="check/init host components + ~/.aisee")
    p.add_argument("--with-models", nargs="*", help="catalog slugs/HF ids to register")
    p.set_defaults(fn=cmd_install)

    p = sub.add_parser("uninstall", help="remove everything AISee put on this host")
    p.add_argument("--keep-cache", action="store_true", help="keep hf-cache/ (weights)")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_uninstall)

    p = sub.add_parser("status", help="one-screen status")
    add_server(p)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("api", help="manage the local REST API daemon")
    p.add_argument("api_cmd", choices=["start", "stop", "status"])
    p.add_argument("--port", type=int, help="bind port (persisted to config.toml)")
    p.add_argument("--host", help="bind address, e.g. 0.0.0.0 (LAN) or 127.0.0.1 "
                                  "(local only); persisted to config.toml")
    p.set_defaults(fn=cmd_api)

    p = sub.add_parser("model", help="install/remove/start/stop models")
    ms = p.add_subparsers(dest="model_cmd", required=True)
    m = ms.add_parser("install")
    m.add_argument("name", help="catalog slug or HF id (org/Model)")
    m.add_argument("--image")
    m.add_argument("--gpu-frac", type=float)
    m.add_argument("--max-model-len", type=int, help="context length (default: auto-sized to the GPU)")
    m.add_argument("--concurrency", type=int, help="parallel inferences for this model (default 3)")
    m.add_argument("--port", type=int)
    m.add_argument("--idle-timeout", type=int)
    m.add_argument("--arg", action="append", help="extra vllm serve arg (repeatable)")
    m.add_argument("--hf-token")
    m = ms.add_parser("remove")
    m.add_argument("slug")
    ms.add_parser("list")
    m = ms.add_parser("start")
    m.add_argument("slug")
    add_server(m)
    m = ms.add_parser("stop")
    m.add_argument("slug")
    add_server(m)
    m = ms.add_parser("logs")
    m.add_argument("slug")
    m.add_argument("-n", type=int, default=60)
    m = ms.add_parser("default")
    m.add_argument("slug")
    for name, sp in ms.choices.items():
        if name in ("list",):
            add_server(sp)
    p.set_defaults(fn=cmd_model)

    p = sub.add_parser("creds", help="manage stored credentials")
    cs = p.add_subparsers(dest="creds_cmd", required=True)
    m = cs.add_parser("set")
    m.add_argument("key")
    m.add_argument("value", nargs="?")
    m = cs.add_parser("unset")
    m.add_argument("key")
    cs.add_parser("list")
    p.set_defaults(fn=cmd_creds)

    def add_query(p, need_q: bool, need_e: bool):
        p.add_argument("media", nargs="+", help="image/video files")
        if need_q:
            p.add_argument("-q", "--question", required=True)
        if need_e:
            p.add_argument("-e", "--expectation", required=True)
        p.add_argument("--model")
        p.add_argument("--frames", type=int)
        p.add_argument("--fps", type=float)
        p.add_argument("--native", action="store_true", help="send video natively (not frames)")
        p.add_argument("--context")
        p.add_argument("--context-file")
        p.add_argument("--max-tokens", type=int)
        p.add_argument("--no-wait", action="store_true", help="print task id and exit")
        add_server(p)

    p = sub.add_parser("look", help="free-form question about media")
    add_query(p, True, False)
    p.set_defaults(fn=lambda a: cmd_query(a, "look"))

    p = sub.add_parser("assert", help="pass/fail expectation about media")
    add_query(p, False, True)
    p.set_defaults(fn=lambda a: cmd_query(a, "assert"))

    p = sub.add_parser("watch", help="chunked whole-video analysis")
    p.add_argument("media", nargs=1, help="video file")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-q", "--question")
    g.add_argument("-e", "--expectation")
    p.add_argument("--model")
    p.add_argument("--fps", type=float)
    p.add_argument("--chunk-seconds", type=float)
    p.add_argument("--scale", type=int)
    p.add_argument("--server-frames", type=int)
    p.add_argument("--context")
    p.add_argument("--context-file")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--no-wait", action="store_true")
    add_server(p)
    p.set_defaults(fn=lambda a: cmd_query(a, "watch"))

    p = sub.add_parser("task", help="inspect the task queue")
    ts = p.add_subparsers(dest="task_cmd", required=True)
    m = ts.add_parser("list")
    m.add_argument("--status")
    m.add_argument("--model")
    m = ts.add_parser("show")
    m.add_argument("id")
    m = ts.add_parser("cancel")
    m.add_argument("id")
    for sp in ts.choices.values():
        add_server(sp)
    p.set_defaults(fn=cmd_task)

    p = sub.add_parser("describe", help="print the agent-facing API guide")
    p.add_argument("--json", action="store_true")
    add_server(p)
    p.set_defaults(fn=cmd_describe)

    p = sub.add_parser("mcp", help="run the MCP server on stdio (consumer capabilities only)")
    p.add_argument("--server", help="API base URL (default: local; or AISEE_SERVER env)")
    p.set_defaults(fn=cmd_mcp)

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except BrokenPipeError:  # stdout consumer (e.g. `| head`) went away
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except (RuntimeError, OSError, ValueError) as e:
        try:
            _p(f"error: {e}")
        except BrokenPipeError:
            pass
        return 2
    except KeyboardInterrupt:
        _p("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
