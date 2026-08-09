# AISee v{{version}} - API guide for AI agents

**AISee is a tool that gives AI agents eyes and ears.** Send it images or video files with a question (`look`), an expectation to verify (`assert`), or a whole video to analyze chunk by chunk (`watch`); it runs a vision-language model on this host and returns the answer. Send it a recording (an audio file, or a video with audio) to `transcribe` (word-timestamped transcript of every audio track/channel; optional per-lane diarization) or `diarize` (who spoke when). Everything is asynchronous: you submit a task and poll it.

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
| GET | `/v1/tasks/{id}/artifacts` | consumer | derived output files (transcripts, RTTM) with sizes |
| GET | `/v1/tasks/{id}/artifacts/{name}` | consumer | download one artifact, e.g. `transcript.srt` |
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
`assert_visual`, `watch`, `transcribe`, `diarize`, `list_models`, `list_tasks`, `get_task`,
`cancel_task`, `describe`, `health`. MCP tool media paths are resolved on this host (no upload);
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
- `transcribe` - word-timestamped transcript of EVERY audio lane of ONE file; with
  `diarize: true` each lane is also speaker-attributed. A lane is one audio track, or one
  CHANNEL of a stereo/multi-channel track (stereo often encodes two separate feeds), always
  processed as independent mono. AISee does NOT interpret or merge lanes (author mic vs
  system audio vs per-participant is the caller's business); the only cross-lane logic is
  factual: lanes with identical audio are marked `duplicate_of` and not transcribed twice.
  Result: `{"tracks": [{label, stream, channel, text, segments: [{start, end, speaker?,
  text}], word_count, rtfx, num_speakers?, speakers?}], "num_tracks", "asr_rtfx",
  "diarized", "artifacts"}`; single-lane results also carry flat
  text/segments/num_speakers for convenience. Timestamps are absolute seconds in the
  recording. Full word timings plus rendered per-lane `transcript[-<lane>].txt/.srt/.vtt`
  are artifacts (`GET /v1/tasks/{id}/artifacts/...`).
- `diarize` - who spoke when, no transcript, per lane. Result: `{"tracks": [{label, turns:
  [{start, end, speaker}], num_speakers, speakers, rtfx}], ...}` (single-lane also flat) +
  per-lane `diarization[-<lane>].rttm` artifacts.

Submission parameters (`POST /v1/tasks`, multipart field `params` as a JSON string, files in
`files`): `kind` (look|assert|watch|transcribe|diarize), `model` (slug; omit for the default),
`question` or
`expectation`, `fps` (video sampling rate: 1 for overviews, 8-15 to hunt flicker/glitches),
`frames` (even-sampled frame count when fps is not set), `native` (send video natively instead of
frames, if the model supports it), `chunk_seconds` (watch), `context` (extra background text the
model should assume), `max_tokens`, `thinking` (bool; for models marked **Thinking: optional** in
the model list below — enables/disables chain-of-thought reasoning; default `true`; has no effect
on always-on reasoning models). Audio kinds: `diarize` (transcribe: also attribute speakers per lane;
default `false`), `min_speakers` / `max_speakers` / `num_speakers` (per-lane diarization hints -
pass them when the count is roughly known; long multi-party audio tends to over-split). A
suspicious diarization (>10 speakers, unhinted) is flagged `suspicious_speaker_count: true`.
Expect roughly realtime/30 or faster for transcription on this class of host (a 1 h recording
in ~2-5 min); diarization adds a similar order. On unified-memory hosts, transcribing LONG
recordings (tens of minutes) while a large vision model is resident can exceed the memory
pool: the audio engine is then killed and the task fails with a clear "engine connection
failed" error while the host stays healthy - retry after the vision model idle-unloads (or
stop it). Short/medium recordings co-reside fine.

Answer budgets (`max_tokens`): when not passed per call, defaults are per kind - `assert` 1024,
`watch` 4096 per chunk (the final cross-chunk synthesis gets the `look` budget, since its
length scales with chunk count), `look` 8192 (dense OCR must never clip content); reasoning models (always-on)
and toggle models with thinking enabled both get 8192 for every kind since thinking counts against
the same budget. Size it to the largest
useful answer - the cap is your runaway bound, so do not blanket-raise it. Truncation is never
silent: an answer that hit the cap ends with `[truncated at N tokens]` and the result carries
`truncated: true` (per chunk and task-level for `watch`); a truncated `assert` fails with a
distinct "verdict truncated" reason. If a large media payload leaves no room for the requested
budget, the budget is automatically shrunk to fit and the result carries
`max_tokens_clamped: true`.

Repetition handling: watch chunk generation runs with a mild repetition penalty (host
config `watch_repetition_penalty`, default 1.1; look/assert keep neutral sampling so
legitimate OCR repetition survives). Degenerate repetition that still occurs is collapsed
after generation with an inline note and surfaced as `deduped: <units removed>`; an A/B
alternation between two contradictory readings is replaced by a low-confidence line and
flagged `unstable: true` - verify unstable readings with a still. Risky video-mode claims
(specific document/window titles, share-state stories) are automatically cross-checked
against a full-resolution still from the same moment (host config `watch_still_checks`,
default 2 per chunk; 0 disables): refuted titles are removed with an inline note, refuted
share claims become an explicit cannot-determine line, the chunk is flagged
`unstable: true`, and each check is recorded under `still_checks` (kind, at_s, confirmed).
Video-mode invention can still slip through; when a claim is surprising, confirm it against
a single extracted frame yourself.

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
  line), spread evenly over the whole clip. A 60 s clip at a 24-frame budget keeps ~2.5 s
  resolution; a 10 min clip drops to one frame per ~25 s.
- `frames` / `fps`: sampled client-side into the model's image budget (its `Image budget:`
  line above), so e.g. 1 fps covers max_images seconds per request.
- **Use `watch` for anything longer than a few minutes**: it splits the video into chunks of
  `server_frames/fps` seconds so every chunk gets the full frame budget, up to 64 chunks per
  call (about 25 min at fps=1 with 24 s chunks - raise `chunk_seconds` or lower `fps` for
  longer clips, trading per-frame resolution or temporal resolution for reach). A full-budget
  chunk is a big request; expect tens of seconds per chunk. The whole watch (all chunks +
  the final synthesis) must finish within the host's request_timeout (default 1 h).
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
