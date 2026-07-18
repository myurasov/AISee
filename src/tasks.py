# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Task store (sqlite), per-model FIFO workers, model lifecycle manager, idle reaper.

All of this lives in the API server process (spec §3): the CLI reaches it over HTTP only.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from . import blobs, catalog, config, creds, dockerctl, media, paths, registry, vlm

# answer budgets applied when the caller does not pass max_tokens: verdict JSON never
# needs much; dense-OCR looks must never clip content; watch is per chunk. Reasoning
# models get the top tier for every kind - thinking counts against the same budget.
BUILTIN_MAX_TOKENS = {"assert": 1024, "look": 8192, "watch": 4096}
REASONING_MAX_TOKENS = 8192


def resolve_max_tokens(kind: str, params: dict, entry: dict, defaults: dict) -> int:
    """Per-call > config max_tokens_<kind> > config global (>0) > built-in per kind."""
    if params.get("max_tokens"):
        return int(params["max_tokens"])
    per_kind = defaults.get(f"max_tokens_{kind}")
    if per_kind:
        return int(per_kind)
    if defaults.get("max_tokens"):  # legacy single knob; 0 (the default) = unset
        return int(defaults["max_tokens"])
    builtin = BUILTIN_MAX_TOKENS.get(kind, 4096)
    return max(builtin, REASONING_MAX_TOKENS) if entry.get("reasoning") else builtin

TERMINAL = ("done", "failed", "canceled")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY, kind TEXT, model TEXT,
  params TEXT, status TEXT, progress TEXT, timings TEXT,
  result TEXT, error TEXT, created REAL, updated REAL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE TABLE IF NOT EXISTS model_usage (slug TEXT PRIMARY KEY, last_used REAL);
"""


class TaskStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)
            try:  # migration for databases created before watch checkpointing
                self._db.execute("ALTER TABLE tasks ADD COLUMN checkpoint TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
            self._db.commit()

    def create(self, kind: str, model: str, params: dict) -> str:
        tid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO tasks (id, kind, model, params, status, progress, timings, "
                "result, error, created, updated) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (tid, kind, model, json.dumps(params), "queued",
                 json.dumps({"step": "queued", "detail": "waiting in queue"}),
                 json.dumps({"queued_at": now}), None, None, now, now))
            self._db.commit()
        return tid

    def _row_to_task(self, r) -> dict:
        timings = json.loads(r["timings"]) if r["timings"] else {}
        if timings.get("finished_at") and timings.get("queued_at"):
            # wall-clock from submission to a terminal state (includes queue + model load)
            timings["total_s"] = round(timings["finished_at"] - timings["queued_at"], 1)
        if timings.get("finished_at") and timings.get("started_at"):
            # processing time only - excludes waiting in the queue (last attempt if requeued)
            timings["active_s"] = round(timings["finished_at"] - timings["started_at"], 1)
        return {
            "id": r["id"], "kind": r["kind"], "model": r["model"],
            "params": json.loads(r["params"]), "status": r["status"],
            "progress": json.loads(r["progress"]) if r["progress"] else {},
            "timings": timings,
            "result": json.loads(r["result"]) if r["result"] else None,
            "error": json.loads(r["error"]) if r["error"] else None,
            "created": r["created"], "updated": r["updated"],
        }

    def get(self, tid: str) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
        return self._row_to_task(r) if r else None

    def list_tasks(self, status: str | None = None, model: str | None = None,
                   limit: int = 100) -> list[dict]:
        q, args = "SELECT * FROM tasks", []
        conds = []
        if status:
            conds.append("status=?")
            args.append(status)
        if model:
            conds.append("model=?")
            args.append(model)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY created DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [self._row_to_task(r) for r in rows]

    def update(self, tid: str, *, status: str | None = None, progress: dict | None = None,
               timing: dict | None = None, result=None, error: dict | None = None) -> None:
        with self._lock:
            r = self._db.execute("SELECT status, timings FROM tasks WHERE id=?",
                                 (tid,)).fetchone()
            if not r:
                return
            if status and r["status"] in TERMINAL:
                # terminal is final: a worker finishing after a cancel must not flip
                # canceled back to done (it checks for cancellation only between steps)
                return
            sets, args = ["updated=?"], [time.time()]
            if status:
                sets.append("status=?")
                args.append(status)
            if progress is not None:
                sets.append("progress=?")
                args.append(json.dumps(progress))
            if timing:
                t = json.loads(r["timings"] or "{}")
                t.update(timing)
                sets.append("timings=?")
                args.append(json.dumps(t))
            if result is not None:
                sets.append("result=?")
                args.append(json.dumps(result))
            if error is not None:
                sets.append("error=?")
                args.append(json.dumps(error))
            args.append(tid)
            self._db.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id=?", args)
            self._db.commit()

    def save_chunk(self, tid: str, plan_key: str, i: int, chunk) -> None:
        """Checkpoint one finished watch chunk so a server restart resumes, not restarts."""
        with self._lock:
            r = self._db.execute("SELECT checkpoint FROM tasks WHERE id=?", (tid,)).fetchone()
            if not r:
                return
            ck = json.loads(r["checkpoint"]) if r["checkpoint"] else {}
            if ck.get("key") != plan_key:
                ck = {"key": plan_key, "chunks": {}}
            ck["chunks"][str(i)] = chunk
            self._db.execute("UPDATE tasks SET checkpoint=?, updated=? WHERE id=?",
                             (json.dumps(ck), time.time(), tid))
            self._db.commit()

    def load_chunks(self, tid: str, plan_key: str) -> dict[int, dict | None]:
        """Previously checkpointed chunks, only if they belong to the same chunk plan."""
        with self._lock:
            r = self._db.execute("SELECT checkpoint FROM tasks WHERE id=?", (tid,)).fetchone()
        ck = json.loads(r["checkpoint"]) if r and r["checkpoint"] else {}
        if ck.get("key") != plan_key:
            return {}
        return {int(k): v for k, v in (ck.get("chunks") or {}).items()}

    def clear_checkpoint(self, tid: str) -> None:
        with self._lock:
            self._db.execute("UPDATE tasks SET checkpoint=NULL, updated=? WHERE id=?",
                             (time.time(), tid))
            self._db.commit()

    def claim_next(self, model: str) -> dict | None:
        """Oldest queued task for `model` -> preparing_media, atomically."""
        with self._lock:
            r = self._db.execute(
                "SELECT * FROM tasks WHERE status='queued' AND model=? ORDER BY created LIMIT 1",
                (model,)).fetchone()
            if not r:
                return None
            self._db.execute(
                "UPDATE tasks SET status='preparing_media', updated=? WHERE id=?",
                (time.time(), r["id"]))
            self._db.commit()
        return self._row_to_task(r)

    def models_with_queued(self) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT model FROM tasks WHERE status='queued'").fetchall()
        return [r["model"] for r in rows]

    def open_count(self, model: str) -> int:
        with self._lock:
            r = self._db.execute(
                "SELECT COUNT(*) c FROM tasks WHERE model=? AND status NOT IN (?,?,?)",
                (model, *TERMINAL)).fetchone()
        return r["c"]

    def touch_model(self, slug: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO model_usage VALUES (?,?) "
                "ON CONFLICT(slug) DO UPDATE SET last_used=excluded.last_used",
                (slug, time.time()))
            self._db.commit()

    def last_used(self, slug: str) -> float | None:
        with self._lock:
            r = self._db.execute("SELECT last_used FROM model_usage WHERE slug=?",
                                 (slug,)).fetchone()
        return r["last_used"] if r else None

    def requeue_stale(self) -> int:
        """Re-queue tasks that were in flight when a previous server died.

        Their worker threads are gone; without this they would show model_loading/
        running forever. Requeued tasks are picked up by the dispatcher again.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT id FROM tasks WHERE status IN ('preparing_media','model_loading','running')"
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                self._db.executemany(
                    "UPDATE tasks SET status='queued', progress=?, updated=? WHERE id=?",
                    [(json.dumps({"step": "queued",
                                  "detail": "requeued after server restart"}),
                      time.time(), i) for i in ids])
                self._db.commit()
        return len(ids)

    def gc(self, retention_days: float) -> int:
        cutoff = time.time() - retention_days * 86400
        with self._lock:
            rows = self._db.execute(
                "SELECT id FROM tasks WHERE updated<? AND status IN (?,?,?)",
                (cutoff, *TERMINAL)).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                self._db.executemany("DELETE FROM tasks WHERE id=?", [(i,) for i in ids])
                self._db.commit()
        for tid in ids:
            shutil.rmtree(paths.media_dir() / tid, ignore_errors=True)
        return len(ids)


class Core:
    """Server-side singleton: model lifecycle + task workers + reaper."""

    def __init__(self):
        paths.ensure_layout()
        self.cfg = config.load()
        self.store = TaskStore(paths.tasks_db())
        self._load_lock = threading.Lock()      # one model may cold-load at a time
        self._model_locks: dict[str, threading.Lock] = {}
        self._workers: dict[str, list[threading.Thread]] = {}
        self._cancel: set[str] = set()
        self._model_loading: dict[str, str] = {}  # slug -> phase note
        self._stop = threading.Event()

    # ---------------- model lifecycle ----------------

    def model_state(self, slug: str) -> str:
        if slug in self._model_loading:
            return "starting"
        st = dockerctl.container_state(slug)
        if st == "running":
            return "running"
        if st == "exited":
            return "failed"
        return "installed"

    def model_view(self, entry: dict) -> dict:
        slug = entry["slug"]
        return {
            "slug": slug, "hf_id": entry["hf_id"], "port": entry["port"],
            "state": self.model_state(slug),
            "idle_timeout": entry.get("idle_timeout"),
            "last_used": self.store.last_used(slug),
            "default": registry.default_model() == slug,
            "supports_native_video": entry.get("supports_native_video", True),
            "gpu_frac": entry.get("gpu_frac"),
            # size for "prefer the bigger running model" heuristics; None off-catalog
            "weights_gib": (catalog.CATALOG.get(slug) or {}).get("weights_gib"),
            "max_model_len": entry.get("max_model_len"),
            "max_images": entry.get("max_images"),
            "video_frames": entry.get("video_frames"),
            "concurrency": entry.get("concurrency", 1),
            # video sampling default: global config unless the model's TOML overrides it
            "fps": entry.get("fps") or float(self.cfg["defaults"]["fps"]),
            "fps_override": entry.get("fps") is not None,
            "image": entry.get("image"),
            "loading_note": self._model_loading.get(slug),
        }

    def check_gpu_capacity(self, entry: dict) -> None:
        """Refuse a start that would oversubscribe the GPU - before any container work.

        Each model reserves gpu_frac of GPU memory at startup; if the running models'
        fractions plus this one exceed 1.0, vLLM would crash-loop on allocation anyway,
        so fail fast with an actionable message instead."""
        others = [e for e in registry.list_installed()
                  if e["slug"] != entry["slug"]
                  and dockerctl.container_state(e["slug"]) == "running"]
        used = sum(float(e.get("gpu_frac") or 0) for e in others)
        need = float(entry.get("gpu_frac") or 0)
        if used + need > 1.0 + 1e-6:
            resident = ", ".join(f"{e['slug']} (gpu_frac {e['gpu_frac']})" for e in others)
            raise RuntimeError(
                f"not starting '{entry['slug']}': it needs gpu_frac {need} but running "
                f"models already reserve {used:.2f} of the GPU ({resident}). Stop one "
                f"first, or reinstall models with smaller --gpu-frac slices to co-locate.")

    def _lock_for(self, slug: str) -> threading.Lock:
        return self._model_locks.setdefault(slug, threading.Lock())

    def ensure_running(self, slug: str, progress=None) -> dict:
        """Start the model if needed and wait until it serves. Returns the registry entry."""
        entry = registry.get(slug)
        if not entry:
            raise RuntimeError(f"model '{slug}' is not installed")
        with self._lock_for(slug):
            if dockerctl.container_state(slug) == "running":
                try:
                    dockerctl.wait_ready(entry, timeout=10)
                    return entry
                except RuntimeError:
                    pass
                # Container up but engine not serving yet: it may still be downloading
                # weights or loading shards from a previous start. Wait it out with the
                # full load timeout - recreating it here would throw that progress away.
                self._model_loading[slug] = "loading weights (existing container)"
                try:
                    def _pw(note):
                        self._model_loading[slug] = note
                        if progress:
                            progress(note)
                    if progress:
                        progress("model is loading (attaching to an in-progress start)")
                    dockerctl.wait_ready(entry, progress=_pw)
                    self.store.touch_model(slug)
                    return entry
                except RuntimeError:
                    pass  # container died or timed out -> full recreate below
                finally:
                    self._model_loading.pop(slug, None)
            self.check_gpu_capacity(entry)  # fail fast before any container work
            hf_token = creds.resolve("HF_TOKEN")
            with self._load_lock:  # admission control: one cold load at a time
                self._model_loading[slug] = "starting container"
                try:
                    if progress:
                        progress("model is loading: starting container")
                    if not dockerctl.image_present(entry["image"]):
                        self._model_loading[slug] = "pulling image"
                        if progress:
                            progress("model is loading: pulling serving image")
                        dockerctl.pull(entry["image"], creds.resolve("NGC_API_KEY"))
                    dockerctl.start_model(entry, hf_token=hf_token)
                    self._model_loading[slug] = "applying image patches"
                    dockerctl.apply_image_patches(entry)
                    self._model_loading[slug] = "loading weights"

                    def _p(note):
                        self._model_loading[slug] = note
                        if progress:
                            progress(note)
                    dockerctl.wait_ready(entry, progress=_p)
                finally:
                    self._model_loading.pop(slug, None)
            self.store.touch_model(slug)
            return entry

    def start_model_async(self, slug: str) -> None:
        # capacity-check synchronously so the API can reject with a clear error
        # instead of a container silently failing in the background
        entry = registry.get(slug)
        if entry and dockerctl.container_state(slug) != "running":
            self.check_gpu_capacity(entry)
        threading.Thread(target=self._safe_ensure, args=(slug,), daemon=True).start()

    def _safe_ensure(self, slug: str) -> None:
        try:
            self.ensure_running(slug)
        except RuntimeError:
            pass  # state surfaces via model_state()/logs

    def stop_model(self, slug: str) -> None:
        dockerctl.stop_model(slug)

    # ---------------- tasks ----------------

    def submit(self, kind: str, model: str | None, params: dict) -> str:
        slug = model or registry.default_model()
        if not slug:
            raise ValueError("no model specified and no default model installed")
        if not registry.get(slug):
            raise ValueError(f"model '{slug}' is not installed")
        return self.store.create(kind, slug, params)

    def cancel(self, tid: str) -> bool:
        t = self.store.get(tid)
        if not t or t["status"] in TERMINAL:
            return False
        if t["status"] == "queued":
            self.store.update(tid, status="canceled",
                              progress={"step": "canceled", "detail": "canceled while queued"})
        else:
            self._cancel.add(tid)  # best-effort: workers check between steps/chunks
        return True

    # ---------------- worker machinery ----------------

    def start_background(self) -> None:
        n = self.store.requeue_stale()
        if n:
            print(f"aisee: requeued {n} task(s) interrupted by a previous shutdown", flush=True)
        threading.Thread(target=self._dispatcher, daemon=True).start()
        threading.Thread(target=self._reaper, daemon=True).start()

    def _dispatcher(self) -> None:
        """Keep up to `concurrency` workers per model while it has queued tasks.

        vLLM batches the concurrent requests server-side (continuous batching), so
        N workers -> N in-flight inferences on one engine.
        """
        while not self._stop.wait(0.5):
            for slug in self.store.models_with_queued():
                entry = registry.get(slug) or {}
                want = max(1, int(entry.get("concurrency", 1)))
                pool = [w for w in self._workers.get(slug, []) if w.is_alive()]
                while len(pool) < min(want, self.store.open_count(slug)):
                    w = threading.Thread(target=self._worker, args=(slug,), daemon=True)
                    pool.append(w)
                    w.start()
                self._workers[slug] = pool

    def _worker(self, slug: str) -> None:
        """FIFO, one inference at a time per model. Exits when its queue drains."""
        idle_polls = 0
        while idle_polls < 20 and not self._stop.is_set():
            task = self.store.claim_next(slug)
            if task is None:
                idle_polls += 1
                time.sleep(0.5)
                continue
            idle_polls = 0
            try:
                self._process(task)
            except Exception as e:  # never kill the worker on a task error
                self.store.update(task["id"], status="failed",
                                  error={"message": str(e)[:2000], "hint": ""})

    def _reaper(self) -> None:
        """Stop containers idle past their timeout; GC old tasks."""
        while not self._stop.wait(30):
            for entry in registry.list_installed():
                slug = entry["slug"]
                timeout = int(entry.get("idle_timeout") or 0)
                if timeout <= 0 or dockerctl.container_state(slug) != "running":
                    continue
                if slug in self._model_loading or self.store.open_count(slug) > 0:
                    continue
                last = self.store.last_used(slug)
                if last is None:
                    self.store.touch_model(slug)  # adopt unknown-running as used-now
                    continue
                if time.time() - last > timeout:
                    dockerctl.stop_model(slug)
            d = self.cfg["defaults"]
            # task_retention_days is the pre-0.6 name; honor it if a host still sets it
            task_ttl_h = (float(d["task_retention_days"]) * 24 if "task_retention_days" in d
                          else float(d.get("task_ttl_hours", 24)))
            self.store.gc(task_ttl_h / 24)
            # tasks keep hardlinked copies of their media, so blob GC is always safe
            blobs.gc(float(d.get("blob_ttl_hours", 24)))

    # ---------------- task processing ----------------

    def _canceled(self, tid: str) -> bool:
        if tid in self._cancel:
            self._cancel.discard(tid)
            self.store.update(tid, status="canceled",
                              progress={"step": "canceled", "detail": "canceled by request"})
            return True
        return False

    def _progress(self, tid: str, step: str, detail: str = "", **extra) -> None:
        self.store.update(tid, progress={"step": step, "detail": detail, **extra})

    def _process(self, task: dict) -> None:
        tid, kind, slug, p = task["id"], task["kind"], task["model"], task["params"]
        d = self.cfg["defaults"]
        started = time.time()
        self.store.update(tid, timing={"started_at": started})
        entry = registry.get(slug)
        if not entry:
            raise RuntimeError(f"model '{slug}' was removed while the task was queued")

        # model lifecycle (may cold-load)
        if self.model_state(slug) != "running":
            self.store.update(tid, status="model_loading")
            self._progress(tid, "model_loading", "model is loading")
            t0 = time.time()
            self.ensure_running(slug, progress=lambda note: self._progress(tid, "model_loading", note))
            self.store.update(tid, timing={"model_load_s": round(time.time() - t0, 1)})
        else:
            # normally instant; if the container turns out to be mid-load, surface it
            def _late(note):
                self.store.update(tid, status="model_loading")
                self._progress(tid, "model_loading", note)
            self.ensure_running(slug, progress=_late)
        if self._canceled(tid):
            return

        work_dir = paths.media_dir() / tid / "derived"
        media_files = p.get("media", [])
        native = bool(p.get("native", False)) and entry.get("supports_native_video", True)
        frames = int(p.get("frames") or d["frames"])
        fps = float(p["fps"]) if p.get("fps") else None
        max_tokens = resolve_max_tokens(kind, p, entry, d)
        timeout = float(d["request_timeout"])
        context = p.get("context") or None

        t0 = time.time()
        if kind in ("look", "assert"):
            self.store.update(tid, status="preparing_media")
            self._progress(tid, "preparing_media",
                           "sampling frames / encoding media" if any(media.is_video(m) for m in media_files)
                           else "encoding images")
            text = (vlm.with_context(p["question"], context) if kind == "look"
                    else vlm.with_context(f"Expectation to verify: {p['expectation']}", context))
            content = media.build_content(media_files, text, frames=frames, fps=fps,
                                          native=native, max_images=entry["max_images"],
                                          work_dir=work_dir)
            prep_s = round(time.time() - t0, 1)
            self.store.update(tid, status="running", timing={"media_prep_s": prep_s})
            self._progress(tid, "running", "inference in progress")
            if self._canceled(tid):
                return
            t1 = time.time()
            if kind == "look":
                answer, meta = vlm.run_look(entry["port"], entry["hf_id"], content,
                                            max_tokens=max_tokens, timeout=timeout)
                if meta.get("finish_reason") == "length":
                    answer += vlm.truncation_marker(meta)
                result = vlm.annotate({"answer": answer}, meta)
            else:
                result = vlm.run_assert(entry["port"], entry["hf_id"], content,
                                        max_tokens=max_tokens, timeout=timeout)
            self.store.update(tid, status="done", result=result,
                              timing={"inference_s": round(time.time() - t1, 1),
                                      "finished_at": time.time()})
            self._progress(tid, "done", "")
        elif kind == "watch":
            self._watch(tid, entry, p, work_dir, fps=fps or float(d["fps"]),
                        max_tokens=max_tokens, timeout=timeout, context=context)
        else:
            raise RuntimeError(f"unknown task kind '{kind}'")
        self.store.touch_model(slug)

    def _watch(self, tid: str, entry: dict, p: dict, work_dir, *, fps: float,
               max_tokens: int, timeout: float, context: str | None) -> None:
        """Chunked whole-video analysis. Result shape follows the query type (spec §9)."""
        question, expectation = p.get("question"), p.get("expectation")
        if (question is None) == (expectation is None):
            raise RuntimeError("watch: pass exactly one of question / expectation")
        media_files = p.get("media", [])
        if not media_files or not media.is_video(media_files[0]):
            raise RuntimeError("watch: the first media file must be a video")
        path = media_files[0]
        dur = media.video_duration(path)
        if not dur:
            raise RuntimeError(f"cannot read duration of {path}")
        native = entry.get("supports_native_video", True) and p.get("native", True)
        server_frames = int(p.get("server_frames") or entry["video_frames"])
        scale = p.get("scale")
        chunk_seconds = p.get("chunk_seconds")
        if not chunk_seconds:
            budget = server_frames if native else entry["max_images"]
            chunk_seconds = max(1.0, budget / fps)
        n = math.ceil(dur / chunk_seconds)
        # a tail shorter than one frame interval (e.g. a 360.125 s video at 60 s chunks)
        # would yield a frameless segment and crash ffmpeg - fold it into the last chunk
        if n > 1 and dur - (n - 1) * chunk_seconds < max(1.0 / fps, 0.5):
            n -= 1
        if n > 64:
            raise RuntimeError(f"watch: {n} chunks > 64 - raise chunk_seconds or lower fps")

        port, hf_id = entry["port"], entry["hf_id"]
        concurrency = max(1, int(entry.get("concurrency", 1)))

        # request_timeout bounds the WHOLE watch task: every chunk and the final synthesis
        # must finish before this deadline, so each inference gets only the time still left
        deadline = time.time() + timeout

        def _remaining(what: str) -> float:
            rem = deadline - time.time()
            if rem <= 1:
                raise RuntimeError(
                    f"watch exceeded request_timeout ({int(timeout)} s) before {what} - "
                    "raise defaults.request_timeout, lower fps, or use chunk_seconds "
                    "to reduce the chunk count")
            return rem

        # per-stage accounting: prep/inference are SUMS across chunks (they overlap in
        # wall time up to `concurrency`); the chunk phase itself is reported as wall time
        stage_s = {"prep": 0.0, "infer": 0.0}
        stage_lock = threading.Lock()

        def _account(kind: str, t0: float) -> float:
            now = time.time()
            with stage_lock:
                stage_s[kind] += now - t0
            return now

        def _do_chunk(i: int) -> dict | None:
            start = i * chunk_seconds
            # the last chunk absorbs any folded-in tail (see the n adjustment above)
            d_s = (dur - start) if i == n - 1 else min(chunk_seconds, dur - start)
            if d_s <= 0.05:
                return None
            rng = f"{start:.1f}s-{min(start + d_s, dur):.1f}s"
            t_prep = time.time()
            seg = media.reencode_segment(path, start, d_s, fps, scale, work_dir,
                                         tag=f"seg{i}")
            try:
                if expectation is not None:
                    text = vlm.with_context(
                        f"Expectation to verify: {expectation} "
                        f"(This clip covers {rng} of a longer video; judge only this span.)", context)
                    content = media.build_content([str(seg)], text, frames=server_frames,
                                                  fps=None, native=native,
                                                  max_images=entry["max_images"],
                                                  work_dir=work_dir / f"c{i}")
                    t_inf = _account("prep", t_prep)
                    r = vlm.run_assert(port, hf_id, content, max_tokens=max_tokens,
                                       timeout=_remaining(f"chunk {rng}"))
                    _account("infer", t_inf)
                    return {"range": rng, **r}
                text = vlm.with_context(
                    f"This clip covers {rng} of a longer video (the clip's 0:00 is "
                    f"{start:.1f}s absolute). {question} Report every time as an ABSOLUTE "
                    f"position in the full video by adding {start:.1f}s to clip-local times.",
                    context)
                content = media.build_content([str(seg)], text, frames=server_frames,
                                              fps=None, native=native,
                                              max_images=entry["max_images"],
                                              work_dir=work_dir / f"c{i}")
                t_inf = _account("prep", t_prep)
                a, meta = vlm.run_look(port, hf_id, content, max_tokens=max_tokens,
                                       timeout=_remaining(f"chunk {rng}"))
                _account("infer", t_inf)
                if meta.get("finish_reason") == "length":
                    a += vlm.truncation_marker(meta)
                return vlm.annotate({"range": rng, "answer": a}, meta)
            finally:
                seg.unlink(missing_ok=True)

        # map: chunks run concurrently up to the model's concurrency (vLLM batches them);
        # results keep chunk order. Finished chunks are checkpointed so a server restart
        # mid-watch (or mid-synthesis) resumes instead of re-paying for every chunk.
        plan_key = (f"{path}|{fps}|{round(chunk_seconds, 3)}|{n}|{native}|{scale}|"
                    f"{max_tokens}|{'assert' if expectation is not None else 'question'}")
        results: dict[int, dict | None] = self.store.load_chunks(tid, plan_key)
        todo = [i for i in range(n) if i not in results]
        self.store.update(tid, status="running")
        self._progress(tid, "running",
                       (f"resuming: {n - len(todo)}/{n} chunks already analyzed, "
                        f"watching the rest (concurrency {concurrency})") if results else
                       f"watching {n} chunks (concurrency {concurrency})",
                       chunk={"i": n - len(todo), "n": n, "t_start": 0.0, "t_end": 0.0})
        done_count = n - len(todo)
        t_chunks = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_do_chunk, i): i for i in todo}
            for fut in as_completed(futures):
                if self._canceled(tid):
                    for f in futures:
                        f.cancel()
                    return
                i = futures[fut]
                results[i] = fut.result()
                self.store.save_chunk(tid, plan_key, i, results[i])
                done_count += 1
                self._progress(tid, "running",
                               f"watched chunk {done_count}/{n}",
                               chunk={"i": done_count, "n": n, "t_start": 0.0, "t_end": 0.0})
        chunks = [results[i] for i in sorted(results) if results[i] is not None]
        self.store.update(tid, timing={
            "chunks_wall_s": round(time.time() - t_chunks, 1),
            "media_prep_s": round(stage_s["prep"], 1),   # summed across chunks
            "inference_s": round(stage_s["infer"], 1),   # summed across chunks
        })

        out = {"mode": "assert" if expectation is not None else "question", "fps": fps,
               "chunk_seconds": round(chunk_seconds, 2), "native": native,
               "duration_s": round(dur, 2), "chunks": chunks}
        # task-level rollups so consumers need not scan chunks for coverage holes
        if any(c.get("truncated") for c in chunks):
            out["truncated"] = True
        if any(c.get("max_tokens_clamped") for c in chunks):
            out["max_tokens_clamped"] = True
        if expectation is not None:
            failing = [c for c in chunks if not c.get("pass")]
            out["pass"] = not failing
            out["failing_ranges"] = [c["range"] for c in failing]
            out["reason"] = ("all chunks satisfied the expectation" if not failing else
                             "; ".join(f'[{c["range"]}] {c.get("reason", "")}' for c in failing)[:800])
        else:
            self._progress(tid, "running", "synthesizing final answer across chunks")
            t_syn = time.time()
            notes = "\n".join(f'[{c["range"]}] {c["answer"]}' for c in chunks)
            answer, meta = vlm.chat(port, hf_id, [{"role": "user", "content": [{"type": "text", "text":
                "These are sequential observations of one continuous video. Synthesize them into a "
                "single coherent answer to the original question. Each observation's [range] prefix "
                "is its ABSOLUTE span in the full video; treat any clip-local times inside an "
                "observation as offset by that range's start. Cite absolute times only. "
                f"Original question: {question}\n\nObservations:\n{notes}"}]}],
                max_tokens=max_tokens, timeout=_remaining("the final synthesis"))
            if meta.get("finish_reason") == "length":
                answer += vlm.truncation_marker(meta)
            out["answer"] = answer
            vlm.annotate(out, meta)
            self.store.update(tid, timing={"synthesis_s": round(time.time() - t_syn, 1)})
        self.store.update(tid, status="done", result=out, timing={"finished_at": time.time()})
        self.store.clear_checkpoint(tid)
        self._progress(tid, "done", "")
