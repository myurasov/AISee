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
| `get_task(task_id)` | poll one task: status, progress, timings, result | task |
| `cancel_task(task_id)` | cancel a queued/running task | confirmation |
| `health()` | liveness + per-model state summary | status |
| `describe()` | this guide | markdown |

Prefer `assert_visual` over `look` whenever you will branch on the outcome: it returns a
strict boolean plus reasoning and concrete evidence.

Optional parameters on the query tools: `model` (slug from the guide below; omit for the
default), `frames` / `fps` (video frame sampling), `native` (send the video itself instead of
sampled frames; video-capable models only), `context` (background the model cannot see in the
pixels), `max_tokens`; `watch` adds `chunk_seconds` and `wait`.

## Behavior to plan around

- **Media paths are resolved on the AISee host itself** (the machine serving this MCP
  endpoint). Pass paths of files that already exist there; to analyze a file from another
  machine, transfer it to the host first (scp/rsync) or use the REST API, which uploads.
- **Query tools block until the answer is ready.** A model that was idle-unloaded reloads in
  ~2-3 minutes; a first-ever use downloads weights and can take tens of minutes. A slow return
  is normal - do not resubmit, that only queues more work.
- **`watch` on long videos takes minutes.** Pass `wait=false` to get `{task_id}` immediately,
  then poll `get_task` every few seconds until `status` is `done` / `failed` / `canceled`;
  `progress` carries a chunk counter.
- **Media budgets are serving config, not model limits** (typically 16 images per request,
  1 video sampled to 64 frames server-side). There is **no maximum video length - only
  temporal resolution**: a `native` video is reduced to the frame budget spread evenly over
  the clip; `watch` chunks the video so every chunk gets the full budget (about an hour of
  video per call at fps=1). fps=1 suits "what happens"; 8-15 hunts flicker/glitches.
- **Some models are stills-only** (they read a video as a single frame) - check `native
  video` in the model guide below before sending video to a non-default model.
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
