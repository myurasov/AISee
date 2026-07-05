---
name: aisee
description: >
  Use AISee to see: verify screenshots and videos, read text off screens, and judge whether a
  UI looks right. AISee serves vision-language models on a GPU host and exposes them through a
  CLI and a REST API. Use this skill whenever a task requires looking at an image or a video
  file: visual verification in e2e tests, OCR, UI checks, or describing what happens in a
  recording.
triggers: ["look at this screenshot", "verify visually", "check the UI", "watch this video",
           "what does the screen show", "visual QA", "aisee"]
---

# Using AISee (a skill for AI agents)

AISee is a tool that gives AI agents eyes. You send it image or video files plus a question or
an expectation; a vision-language model (VLM) running on a GPU host answers. Everything is
asynchronous: you submit a task, poll it, and read the result.

## The three query kinds - pick the right one

| Kind | Input | Returns | Use when |
|---|---|---|---|
| `look` | media + question | free text | OCR, descriptions, "where is X", open questions |
| `assert` | media + expectation | `{pass, reason, evidence}` | you need a machine-checkable verdict (tests, gates) |
| `watch` | one video + question or expectation | per-chunk results + synthesis / failing time ranges | videos longer than ~1 minute, or time-localized checks |

Prefer `assert` over `look` whenever you will branch on the outcome: it returns a strict
boolean plus the model's reasoning and concrete evidence, and the CLI exit code follows the
verdict (0 pass, 1 fail).

## Setup (once per GPU host)

Prerequisites: Linux, NVIDIA GPU, docker + NVIDIA Container Toolkit, ffmpeg, Python 3.12+, uv.

```bash
git clone https://github.com/myurasov/AISee ~/aisee && cd ~/aisee
uv sync                          # or skip: ./aisee bootstraps its own .venv
./aisee install                  # verifies docker/GPU/ffmpeg, creates ~/.aisee
./aisee creds set HF_TOKEN       # needed for gated models (account must accept model licenses)
./aisee model install qwen3-vl-30b-a3b-instruct   # recommended default model
./aisee api start                # REST API on 0.0.0.0:8484 (configurable)
```

`model install` only registers the model; weights (tens of GB) download on the first start or
first query - that task will report `model_loading` for many minutes. This is normal.
Serving settings (GPU memory fraction, context length, media budgets) are computed
automatically for the detected GPU; override per model with flags if needed.

Optional auth: `./aisee creds set AISEE_API_TOKEN` + restart; then send
`Authorization: Bearer <token>` on every call (only `/`, `/v1/describe`, `/v1/health` stay open).

## Using the CLI

The `aisee` command is a thin client of the REST API and also works from any remote machine
with `--server http://HOST:PORT` or `AISEE_SERVER` set; media files are uploaded automatically.

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
video itself, video-capable models only), `--no-wait` (print task id, poll later).

## Using the REST API

Base `http://HOST:PORT/v1`. Discover everything at runtime: `GET /v1/describe` returns an
agent-oriented guide including the installed models with strengths/weaknesses/pitfalls and
live serving configuration - read it before choosing models or parameters.

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
splits `model_load_s` / `media_prep_s` / `inference_s`. On `failed`, read `error.message`.

Other endpoints: `GET /v1/models`, `POST /v1/models/{slug}/start|stop`, `POST /v1/models`
(install, `{"name": "<catalog slug or HF id>"}`), `DELETE /v1/models/{slug}`, `GET /v1/catalog`,
`GET /v1/gpu` (live utilization/memory/power), `GET /v1/health`. A human-friendly single-file
web console is served at `/`.

## Behavior you must plan around

- **Non-blocking + queued.** Submitting returns immediately. Each model runs a limited number
  of inferences in parallel (default 3); excess tasks queue FIFO. Never assume instant results.
- **`model_loading` can take minutes.** First-ever use downloads weights (10-60+ min);
  a model idle-unloaded (default after 15 min) reloads from cache in ~2-3 min. Keep polling -
  `progress.detail` explains what is happening. Do not resubmit; that just queues more work.
- **Media budgets are serving config, not model limits** (defaults: 16 images per request,
  1 video sampled to 64 frames server-side, sized for a 128k context). Each image/frame costs
  roughly 1-2.5k tokens.
- **There is no maximum video length - only temporal resolution.** A `native` video is reduced
  to the frame budget spread evenly over the clip. For anything longer than a few minutes use
  `watch`: it chunks the video so every chunk gets the full frame budget (about an hour of
  video per call at fps=1). High fps hunts flicker/glitches; fps=1 is enough for "what happens".
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

- One GPU per host; models fit only if weights + KV cache fit the GPU (install warns when they
  cannot). Only models the serving vLLM supports as multimodal chat models work.
- File-based media only (images, video files) - no live streams, no URLs; upload the bytes.
- No MCP server mode; integrate via CLI or REST.
- Results depend on a VLM: it can be wrong, especially on tiny text, precise counts, and exact
  numbers. Design checks so a false verdict is caught (negative controls, evidence review).
- The API server must run on the GPU host itself; clients can be anywhere.

## Quick diagnostic recipes

- API down? `aisee status` on the host; `aisee api start`.
- Task stuck in `model_loading`? Check `aisee model logs <slug>` - usually a weight download.
- `failed` with a memory error: another model or process holds the GPU; stop it and retry.
- HF 403: the token's HF account has not accepted that model's license on its page.
- After updating AISee source: `uv sync` and restart the API (a running daemon keeps old code).
