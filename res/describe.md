# AISee v{{version}} - API guide for AI agents

**AISee is a tool that gives AI agents eyes.** Send it images or video files with a question (`look`), an expectation to verify (`assert`), or a whole video to analyze chunk by chunk (`watch`); it runs a vision-language model on this host and returns the answer. Everything is asynchronous: you submit a task and poll it.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/describe` | this document (markdown; ?format=json for structured) |
| GET | `/v1/health` | liveness + per-model state summary |
| GET | `/v1/models` | installed models: state, port, idle_timeout, default flag |
| POST | `/v1/models/{slug}/start` | start a model (non-blocking; poll /v1/models) |
| POST | `/v1/models/{slug}/stop` | stop a model (frees GPU memory; stays installed) |
| POST | `/v1/tasks` | submit a query -> {id} (multipart: files + params JSON field; or JSON with media_paths on the server host) |
| GET | `/v1/tasks` | list tasks (?status=&model=) |
| GET | `/v1/tasks/{id}` | full task: status, progress, timings, result |
| DELETE | `/v1/tasks/{id}` | cancel a task |

## Task lifecycle (how to use this API)

1. `POST /v1/tasks` - returns `{"id": "..."}` immediately (non-blocking).
2. Poll `GET /v1/tasks/{id}` every 2-5 s. `status` walks through:
   `queued -> preparing_media -> model_loading (only if the model is cold) -> running -> done`
   (`failed` / `canceled` are terminal too). `progress` holds a human-readable `step` + `detail`,
   and for `watch` a `chunk: {i, n, t_start, t_end}` counter.
3. **`model_loading` can take minutes** (cold model start; the largest models take ~9 minutes on
   first load). This is normal - keep polling; `progress.detail` explains what is happening.
4. Read `result` when `status == "done"`; on `failed`, `error.message` says why.

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

## Models installed on this host

{{models}}

## Practical limits

- Image inputs are capped per model (typically 8 per request) - sample videos accordingly.
- Prefer `assert` over `look` when you need a machine-checkable verdict.
- Pass `context` for domain knowledge the model can't see ("the left panel is the scene tree").
- One inference runs at a time per model; tasks queue FIFO. Idle models are auto-stopped after
  their idle timeout and transparently restarted on the next task (expect `model_loading`).
