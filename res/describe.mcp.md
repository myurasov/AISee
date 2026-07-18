# AISee v{{version}} - MCP tool guide

**AISee is a tool that gives AI agents eyes.** You are connected to it over MCP: the tools
below send image/video files to a vision-language model (VLM) running on the AISee host and
return its answers. This guide covers only what you can do from here; the underlying REST API
and host administration are out of scope for an MCP client.

## Tools

| Tool | Use for | Returns |
|---|---|---|
| `look(media, question, ...)` | OCR, descriptions, "where is X", open questions | `{answer}` |
| `assert_visual(media, expectation, ...)` | machine-checkable verdicts (tests, gates) | `{pass, reason, evidence}` |
| `watch(video, question OR expectation, ...)` | whole-video analysis; videos longer than ~1 min | per-chunk results + synthesized `answer`, or `{pass, failing_ranges}` |
| `list_models()` | what is installed and its live state | model list |
| `list_tasks(status?, model?)` | recent/queued tasks | task list |
| `get_task(task_id)` | poll one task: status, progress, timings (incl. `total_s` wall-clock once finished), result | task |
| `cancel_task(task_id)` | cancel a queued/running task | confirmation |
| `health()` | liveness + per-model state summary | status |
| `describe()` | this guide | markdown |

Prefer `assert_visual` over `look` whenever you will branch on the outcome: it returns a
strict boolean plus reasoning and concrete evidence.

Optional parameters on the query tools: `model` (slug from the guide below; omit for the
default), `frames` / `fps` (video frame sampling), `native` (send the video itself instead of
sampled frames; video-capable models only), `context` (background the model cannot see in the
pixels), `max_tokens`; `watch` adds `chunk_seconds` and `wait`.

## Uploading media from your machine

MCP tool calls carry no file bytes, so a media entry is either a **path on the AISee host**
or a **`sha256:<hex>` reference** to content you upload first over HTTP to the same server
(one-time per unique file; the store keeps it for a TTL, default 24 hours, refreshed on
each reuse).

**Access for this host: HTTP API base `{{api_base}}`; consumer token: `{{consumer_token}}`**
(send it as `Authorization: Bearer <token>`; omit the header if auth is disabled).

1. Compute the SHA-256 of the file bytes (lowercase hex): `sha256sum f.png` (Linux),
   `shasum -a 256 f.png` (macOS), or `hashlib.sha256(open("f","rb").read()).hexdigest()`.
2. Probe `GET {{api_base}}/v1/blobs/<sha256>` -> `{"exists": ...}`. If it exists, skip the
   upload.
3. Upload if missing: `curl -s -X POST {{api_base}}/v1/blobs -H "Authorization: Bearer
   {{consumer_token}}" -F files=@f.png` -> `[{"sha256": "...", "size": ...}]`.
4. Pass `"sha256:<hex>"` as the media entry to `look` / `assert_visual` / `watch`.

## Behavior to plan around

- **Media entries resolve on the AISee host.** Host paths must already exist there;
  for files on your machine use the `sha256:` upload recipe above (or scp/rsync the file
  to the host).
- **Query tools block until the answer is ready.** A model that was idle-unloaded reloads in
  ~2-3 minutes; a first-ever use downloads weights and can take tens of minutes. A slow return
  is normal - do not resubmit, that only queues more work.
- **`watch` on long videos takes minutes.** Pass `wait=false` to get `{task_id}` immediately,
  then poll `get_task` every few seconds until `status` is `done` / `failed` / `canceled`;
  `progress` carries a chunk counter. The whole watch (all chunks + synthesis) must finish
  within the host's request_timeout (default 1 h).
- **Media budgets are serving config, not model limits** (per-model `max_images` is sized
  so a full batch of 1080p stills fills the context - see each model's `Image budget:`
  line below; 1 video sampled to 24 frames server-side - 24 keeps each frame at ~720p,
  since the video pixel budget is shared across frames). There is **no maximum video length - only temporal
  resolution**: a `native` video is reduced to the frame budget spread evenly over the clip;
  `watch` chunks the video so every chunk gets the full budget - chunk length is frame
  budget / fps (24 s per chunk at fps=1; sparser fps means longer chunks), up to 64 chunks
  (~25 min at fps=1) per call - raise `chunk_seconds` or lower `fps` for longer clips.
  Chunks queue within one call, expect tens of seconds each. fps=1 suits "what happens";
  8-15 hunts flicker/glitches.
- **Some models are stills-only** (they read a video as a single frame) - check `native
  video` in the model guide below before sending video to a non-default model.
- **Spatial resolution**: AISee sends media at source resolution (`look` extracts
  native-resolution frames; the only AISee-side downscale is the optional `scale` param on
  `watch`) - the model's preprocessor is the only implicit resizer. Each model's
  `Input resolution:` line below gives the exact still and per-video-frame pixel budgets;
  when fine text must survive (dense OCR), prefer a full-res still via `look` over a
  video frame.
- **Answer budgets are per kind and truncation is never silent.** Without an explicit
  `max_tokens`: `assert_visual` 1024, `watch` 4096 per chunk, `look` 8192; reasoning models
  8192 everywhere (thinking counts against the same budget). A capped answer ends with
  `[truncated at N tokens]` and carries `truncated: true`; a truncated assert fails with a
  "verdict truncated" reason; `max_tokens_clamped: true` means a large media payload forced
  a smaller budget. Size `max_tokens` to the largest useful answer - it is a runaway bound,
  not a target.
- **Repetition and stability flags.** Degenerate repetition in answers is collapsed
  post-hoc (`deduped: N`); an A/B alternation between contradictory readings becomes one
  low-confidence line flagged `unstable: true` - verify with a still frame. Video-mode
  answers can invent plausible content; confirm surprising claims against an extracted
  still.
- **Trust but verify verdicts.** When an `assert_visual` verdict is surprising, read its
  `reason`/`evidence` and consider a follow-up `look` before acting on it.
- **Model management is not available over MCP** (consumer capabilities only). If a model you
  need is not installed or will not start, ask the host operator.

## Models available on this host

Each entry shows its live state and serving configuration (context window, per-request media
budgets, concurrency). Consult strengths/weaknesses/pitfalls before choosing a non-default
model.

{{models}}

## Tips

- Pass `context` for domain knowledge the model cannot see ("the left panel is the scene tree").
- Results come from a VLM: it can be wrong on tiny text, precise counts, and exact numbers.
  Design checks so a false verdict gets caught (negative controls, evidence review).
- Tasks run in parallel up to each model's `concurrent inferences` setting; excess queues FIFO.
  Idle models auto-stop and transparently restart on the next query (expect a slow first call).
