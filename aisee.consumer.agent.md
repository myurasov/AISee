---
name: aisee-consumer
description: >
  Use AISee to see: verify screenshots and videos, read text off screens, and judge whether a
  UI looks right. AISee serves vision-language models on a GPU host and exposes them through a
  CLI, a REST API, and an MCP server. Adopt this agent whenever a task requires looking at an
  image or a video file: visual verification in e2e tests, OCR, UI checks, or describing what
  happens in a recording. This is the CONSUMER role: querying only; for installing AISee or
  managing models, see aisee.admin.agent.md.
triggers: ["look at this screenshot", "verify visually", "check the UI", "watch this video",
           "what does the screen show", "visual QA", "aisee"]
---

# AISee consumer agent (for AI agents that query AISee)

AISee is a tool that gives AI agents eyes. You send it image or video files plus a question or
an expectation; a vision-language model (VLM) running on a GPU host answers. Everything is
asynchronous: you submit a task, poll it, and read the result. As a consumer you query and
inspect - you do not manage models or the server (those are admin actions and, if the host is
configured with an admin token, will answer 403 to you).

## The three query kinds - pick the right one

| Kind | Input | Returns | Use when |
|---|---|---|---|
| `look` | media + question | free text | OCR, descriptions, "where is X", open questions |
| `assert` | media + expectation | `{pass, reason, evidence}` | you need a machine-checkable verdict (tests, gates) |
| `watch` | one video + question or expectation | per-chunk results + synthesis / failing time ranges | videos longer than ~1 minute, or time-localized checks |

Prefer `assert` over `look` whenever you will branch on the outcome: it returns a strict
boolean plus the model's reasoning and concrete evidence, and the CLI exit code follows the
verdict (0 pass, 1 fail).

## Access

You need the server URL (`http://HOST:PORT`, default port 8484) and, if the host requires
auth, the **consumer token** (`AISEE_API_TOKEN`). Get both from whoever operates the host
(or from the admin agent). Set them once:

```bash
export AISEE_SERVER=http://HOST:PORT
export AISEE_API_TOKEN=<consumer token>       # only if the host requires it
```

Three equivalent ways in, all backed by the same REST API:

1. **CLI** - `aisee <cmd>` from the source checkout (or `./aisee` at its root); works from any
   machine, media files are uploaded automatically.
2. **REST** - plain HTTP; send `Authorization: Bearer <consumer token>` when auth is on.
3. **MCP** - the API server speaks MCP (streamable HTTP) at `http://HOST:PORT/mcp`; nothing
   to install. It exposes exactly these consumer capabilities as tools (`look`,
   `assert_visual`, `watch`, `list_models`, `list_tasks`, `get_task`, `cancel_task`,
   `describe`, `health`). Register it in your harness, e.g.:

   ```json
   {"mcpServers": {"aisee": {"type": "http", "url": "http://HOST:PORT/mcp",
                             "headers": {"Authorization": "Bearer <consumer token>"}}}}
   ```

   The MCP endpoint carries consumer capabilities only, so it cannot manage models by
   design. Query tools block until the answer is ready; for long `watch` jobs pass
   `wait=false` and poll `get_task`. **MCP media entries resolve on the AISee host**: pass
   a host path, or upload your local file once over HTTP and pass a `sha256:` reference:

   ```bash
   sha=$(shasum -a 256 shot.png | cut -d' ' -f1)          # sha256sum on Linux
   curl -s http://HOST:PORT/v1/blobs/$sha                  # {"exists": ...} - probe first
   curl -s -X POST http://HOST:PORT/v1/blobs \
        -H "Authorization: Bearer <consumer token>" -F files=@shot.png
   # then call e.g. assert_visual(media=["sha256:<sha>"], expectation=...)
   ```

## Using the CLI

```bash
aisee look shot.png -q "What error message is shown?"
aisee assert shot.png -e "the Start button is visible and enabled"     # exit code = verdict
aisee assert run.mp4 -e "the app reaches the main menu" --native
aisee watch run.mp4 -q "describe what the user does" --fps 2
aisee watch run.mp4 -e "the frame counter increases monotonically" --fps 8
aisee status | model list | task list | task show <id>
```

Useful flags: `--model <slug>` (else the default model), `--context "<background the model
cannot see in pixels>"`, `--frames N` / `--fps R` (video frame sampling), `--native` (send the
video itself, video-capable models only), `--no-wait` (print task id, poll later),
`--server URL`, `--token T`.

## Using the REST API

Base `http://HOST:PORT/v1`. Discover everything at runtime: `GET /v1/describe` (no auth
needed) returns an agent-oriented guide including the installed models with
strengths/weaknesses/pitfalls and live serving configuration - read it before choosing models
or parameters.

Submit and poll:

```
POST /v1/tasks     multipart: files=<media>..., params=<JSON string>
                   params: {"kind":"look|assert|watch", "question"|"expectation":"...",
                            "model":"<slug>?", "fps"?, "frames"?, "native"?, "chunk_seconds"?,
                            "context"?, "max_tokens"?}
  -> {"id": "..."}
GET  /v1/tasks/{id}   poll every 2-5 s until status is done | failed | canceled
```

Task statuses walk `queued -> preparing_media -> model_loading (cold model only) -> running ->
done`. `progress` carries a human-readable step, and for `watch` a chunk counter. `timings`
splits `model_load_s` / `media_prep_s` / `inference_s` and, once terminal, includes `total_s`
(wall-clock seconds from submission to finish). On `failed`, read `error.message`.

**Upload dedup:** uploads are content-addressed by the SHA-256 of the file bytes and kept
for a TTL (default 24 h, refreshed on reuse). Probe `GET /v1/blobs/{sha256}` (hash via `sha256sum` /
`shasum -a 256` / python `hashlib.sha256(data).hexdigest()`); if it exists, pass
`"sha256:<hash>"` as the media entry (`media_paths` in JSON, or an ordered `media` list in
the multipart `params`) instead of re-uploading. `POST /v1/blobs` uploads without creating a
task. The CLI and web console negotiate this automatically - repeat submissions of the same
file skip the upload.

Consumer endpoints: `GET /v1/models`, `GET /v1/catalog`, `GET /v1/gpu`, `GET /v1/health`,
`GET/POST /v1/tasks`, `GET/DELETE /v1/tasks/{id}`. A human-friendly web console is served
at `/`. Model management (`POST /v1/models`, `DELETE /v1/models/{slug}`,
`POST /v1/models/{slug}/start|stop`) requires the admin token - not your role.

## Behavior you must plan around

- **Non-blocking + queued.** Submitting returns immediately. Each model runs a limited number
  of inferences in parallel (default 3); excess tasks queue FIFO. Never assume instant results.
- **`model_loading` can take minutes.** First-ever use downloads weights (10-60+ min);
  a model idle-unloaded (default after 15 min) reloads from cache in ~2-3 min. Keep polling -
  `progress.detail` explains what is happening. Do not resubmit; that just queues more work.
- **Media budgets are serving config, not model limits** (defaults: 16 images per request,
  1 video sampled to 24 frames server-side - 24 keeps each frame at ~720p, since the video
  pixel budget is shared across frames). Each image/frame costs roughly 1-2.5k tokens.
- **There is no maximum video length - only temporal resolution.** A `native` video is reduced
  to the frame budget spread evenly over the clip. For anything longer than a few minutes use
  `watch`: it chunks the video so every chunk gets the full frame budget - chunk length is
  frame budget / fps (24 s per chunk at fps=1; sparser fps means longer chunks), up to 64
  chunks (~25 min at fps=1) per call - raise `chunk_seconds` or lower `fps` for longer clips.
  Chunks queue within one call. High fps hunts flicker/glitches; fps=1 is enough for "what
  happens".
- **Some models are stills-only** (they read a video as a single frame). Check `native video`
  in `/v1/describe` before sending video to a non-default model.
- **Model choice matters.** The default (Qwen3-VL MoE) is the safe all-rounder: correct OCR,
  video, ~1-4 s per still. Specialists exist for UI element grounding, temporal/physical video
  reasoning, and minimal GPU footprint; one known model reads dense numbers unreliably. Always
  consult the model guide in `/v1/describe` - it states each model's measured strengths,
  weaknesses, and pitfalls.
- **Trust but verify verdicts.** `assert` returns `evidence`; when a verdict is surprising,
  read `reason`/`evidence` and consider a follow-up `look` before acting on it.

## Limitations

- File-based media only (images, video files) - no live streams, no URLs; upload the bytes.
- Results depend on a VLM: it can be wrong, especially on tiny text, precise counts, and exact
  numbers. Design checks so a false verdict is caught (negative controls, evidence review).
- A model that is not installed cannot be queried; `GET /v1/catalog` shows what could be
  installed, but installing is an admin action - ask the operator / admin agent.
- 401 means your token is missing or wrong; 403 means the action needs the admin token.

## Quick diagnostic recipes

- Connection refused? The API daemon is down on the host - an admin must start it.
- Task stuck in `model_loading`? Almost always a weight download - keep polling `progress`.
- `failed` with a memory error: the GPU is busy; retry later or tell the operator.
- Verdict looks wrong? Read `evidence`, retry with `--context` or a different model.
