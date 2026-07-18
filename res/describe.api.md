# AISee v{{version}} - API guide for AI agents

**AISee is a tool that gives AI agents eyes.** Send it images or video files with a question (`look`), an expectation to verify (`assert`), or a whole video to analyze chunk by chunk (`watch`); it runs a vision-language model on this host and returns the answer. Everything is asynchronous: you submit a task and poll it.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/v1/describe` | open | this document (markdown; ?format=json for structured) |
| GET | `/v1/health` | open | liveness + per-model state summary |
| GET | `/v1/gpu` | consumer | live GPU stats: utilization, memory, power, temperature |
| GET | `/v1/models` | consumer | installed models: state, port, idle_timeout, default flag |
| GET | `/v1/catalog` | consumer | built-in model catalog with installed flags |
| GET | `/v1/config` | consumer | effective global configuration (api + defaults) |
| POST | `/v1/tasks` | consumer | submit a query -> {id} (multipart: files + params JSON field; or JSON with media_paths on the server host) |
| GET | `/v1/tasks` | consumer | list tasks (?status=&model=) |
| GET | `/v1/tasks/{id}` | consumer | full task: status, progress, timings, result |
| DELETE | `/v1/tasks/{id}` | consumer | cancel a task |
| GET | `/v1/tasks/{id}/media` | consumer | the task's media files with facts: kind, dimensions, duration, frames, size |
| GET | `/v1/tasks/{id}/media/{i}` | consumer | download the task's i-th media file; append `/thumb` for a JPEG thumbnail |
| GET | `/v1/blobs/{sha256}` | consumer | dedup probe: {exists, size} for already-uploaded content |
| POST | `/v1/blobs` | consumer | upload media into the content store -> [{sha256, size}] |
| POST | `/v1/models` | admin | install a model: {"name": catalog slug or HF id} |
| DELETE | `/v1/models/{slug}` | admin | uninstall (weights stay cached) |
| POST | `/v1/models/{slug}/start` | admin | start a model (non-blocking; poll /v1/models; 409 if it would oversubscribe GPU memory) |
| POST | `/v1/models/{slug}/stop` | admin | stop a model (frees GPU memory; stays installed) |

## Authentication

Auth is optional and off unless the host sets tokens. Send `Authorization: Bearer <token>`.

- **consumer** endpoints require the consumer token (`AISEE_API_TOKEN`) when it is set on the
  host; the admin token is accepted there too. Without a consumer token on the host they are
  open.
- **admin** endpoints (model management) require the admin token (`AISEE_ADMIN_TOKEN`) when
  set; a valid consumer token gets **403** there, a missing/wrong token gets **401**. If only
  the consumer token is set on the host, it guards everything (single-token mode).
- `open` endpoints never need a token.

If you were given one token, it is almost certainly the consumer token: you can query and
inspect, but not install/start/stop models - ask the host operator for those.

This API is also exposed as an MCP server (streamable HTTP) at `/mcp` on the same
host/port, guarded by the consumer token, with consumer capabilities only: tools `look`,
`assert_visual`, `watch`, `list_models`, `list_tasks`, `get_task`, `cancel_task`,
`describe`, `health`. MCP tool media paths are resolved on this host (no upload);
`GET /v1/describe?flavor=mcp` returns the MCP-specific guide.

## Task lifecycle (how to use this API)

1. `POST /v1/tasks` - returns `{"id": "..."}` immediately (non-blocking).
2. Poll `GET /v1/tasks/{id}` every 2-5 s. `status` walks through:
   `queued -> preparing_media -> model_loading (only if the model is cold) -> running -> done`
   (`failed` / `canceled` are terminal too). `progress` holds a human-readable `step` + `detail`,
   and for `watch` a `chunk: {i, n, t_start, t_end}` counter.
3. **`model_loading` can take minutes** (cold model start; the largest models take ~9 minutes on
   first load). This is normal - keep polling; `progress.detail` explains what is happening.
4. Read `result` when `status == "done"`; on `failed`, `error.message` says why.
   `timings` breaks the run down (`model_load_s`, `media_prep_s`, `inference_s`) and, once
   terminal, includes `total_s` - the wall-clock seconds from submission to finish.

Task kinds and their `result` shapes:
- `look` - free-form question about the media. Result: `{"answer": "<text>"}`.
- `assert` - pass/fail judgment of an `expectation`. Result:
  `{"pass": bool, "reason": str, "evidence": str}`. Use for visual regression / e2e checks.
- `watch` - chunked whole-video analysis. With `expectation`: per-chunk verdicts +
  `{"pass": bool, "failing_ranges": [...]}` (timestamps where it broke). With `question`:
  per-chunk findings + a synthesized `answer` over the whole video.

Submission parameters (`POST /v1/tasks`, multipart field `params` as a JSON string, files in
`files`): `kind` (look|assert|watch), `model` (slug; omit for the default), `question` or
`expectation`, `fps` (video sampling rate: 1 for overviews, 8-15 to hunt flicker/glitches),
`frames` (even-sampled frame count when fps is not set), `native` (send video natively instead of
frames, if the model supports it), `chunk_seconds` (watch), `context` (extra background text the
model should assume), `max_tokens`.

## Example

```
curl -s -X POST http://HOST:PORT/v1/tasks \
  -F files=@screenshot.png \
  -F 'params={"kind":"assert","expectation":"the Start button is visible and enabled"}'
# -> {"id":"3f2a..."}; then poll:
curl -s http://HOST:PORT/v1/tasks/3f2a...
```

## Upload dedup (skip re-sending media the server already has)

The server keeps uploaded media in a content-addressed store keyed by the SHA-256 of the
file bytes. Blobs live for a configurable TTL (default 24 hours; each reuse refreshes it),
so recently sent content never needs re-uploading:

1. Compute the hash of the file bytes (lowercase hex, 64 chars):
   - shell: `sha256sum file.mp4` (Linux) or `shasum -a 256 file.mp4` (macOS)
   - python: `hashlib.sha256(open("f","rb").read()).hexdigest()`
   - node: `crypto.createHash("sha256").update(buf).digest("hex")`
2. Probe: `GET /v1/blobs/{sha256}` -> `{"exists": true|false, "size": ...}`.
3. If it exists, reference it instead of uploading: use `"sha256:<hash>"` as a media entry -
   in the JSON submission's `media_paths` list, or in an optional ordered `media` list
   inside the multipart `params` (entries are either `sha256:` refs or the filenames of the
   files you do upload). Order is preserved.
4. If it does not exist, upload as usual - every uploaded file enters the store
   automatically, so the same bytes are skippable next time. `POST /v1/blobs`
   (multipart `files`) uploads without creating a task.

```
sha=$(sha256sum run.mp4 | cut -d' ' -f1)
curl -s http://HOST:PORT/v1/blobs/$sha                     # {"exists": true, ...}
curl -s -X POST http://HOST:PORT/v1/tasks -H 'Content-Type: application/json' \
  -d "{\"kind\":\"watch\",\"question\":\"what happens?\",\"media_paths\":[\"sha256:$sha\"]}"
```

The `aisee` CLI and the web console do this negotiation automatically.

## Models installed on this host

Each entry shows its live state and its **serving configuration** (context window and
per-request media budgets - these are deployment settings, not model limits; they are sized so
a request fits the context window and the KV cache fits the model's GPU slice).

{{models}}

## Video length and sampling

There is **no hard maximum video length** - only temporal resolution:

- `native`: the video is reduced server-side to the model's frame budget (see its Serving
  line), spread evenly over the whole clip. A 60 s clip at a 64-frame budget keeps ~1 s
  resolution; a 10 min clip drops to one frame per ~9 s.
- `frames` / `fps`: sampled client-side into the image budget (16 by default), so e.g. 1 fps
  covers 16 s per request.
- **Use `watch` for anything longer than a few minutes**: it splits the video into chunks of
  `server_frames/fps` seconds so every chunk gets the full frame budget, up to 64 chunks per
  call (about an hour at fps=1 with 64 s chunks; at fps=15 chunks are ~4 s - raise
  `chunk_seconds` or lower `fps` for long clips). A full-budget chunk is a big request; expect
  tens of seconds per chunk.
- Stills-only models (native video: no in the guide above) read a clip as a single frame - use
  frame sampling or pick a video-capable model.
- **Spatial resolution**: AISee sends media at source resolution - `look` extracts
  native-resolution frames, and the only AISee-side downscale is the optional `scale` task
  param on `watch`. The model's preprocessor is the only implicit resizer; each model's
  `Input resolution:` line above gives the exact still and per-video-frame pixel budgets, so
  check it before relying on small text (OCR of fine print may need a full-res still via
  `look` instead of a video frame).

## Tips

- Prefer `assert` over `look` when you need a machine-checkable verdict.
- Pass `context` for domain knowledge the model can't see ("the left panel is the scene tree").
- Up to a model's `concurrent inferences` setting (see its Serving line) run in parallel;
  further tasks queue FIFO. `watch` chunks use the same parallelism internally. Idle models are
  auto-stopped after their idle timeout and transparently restarted on the next task (expect
  `model_loading`).
