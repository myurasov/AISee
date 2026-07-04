# AISee <!-- omit in toc -->

- [What Is AISee?](#what-is-aisee)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)
- [Quick Start](#quick-start)
- [Installing Models](#installing-models)
  - [Built-In Catalog](#built-in-catalog)
  - [Other Models](#other-models)
- [REST API](#rest-api)
  - [Server](#server)
  - [Reference](#reference)
- [Credentials](#credentials)
- [Local Data](#local-data)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## What Is AISee?

AISee is a tool that gives AI agents eyes. It serves vision-language models in docker containers
on a GPU host and answers questions about images and video files, over a CLI or a REST API.

There are three kinds of queries:

- `look` - free-form question, returns text. OCR, descriptions, "where is the button".
- `assert` - an expectation to verify, returns `{pass, reason, evidence}`. Meant for visual
  regression and e2e checks; the CLI exit code follows the verdict.
- `watch` - whole-video analysis, chunk by chunk, at a chosen fps. Given an expectation it
  returns per-chunk verdicts and `failing_ranges` (the time spans where it broke); given a
  question it returns per-chunk notes and a synthesized answer for the whole video.

The CLI is a thin client of the REST API, so there is a single code path: anything the CLI does
can be done with curl, and both feed the same task queue. All queries are asynchronous - you get
a task id back and poll it; progress like "model is loading" or "chunk 3/12" is visible per task.

## Installation

### Prerequisites

- Linux GPU host
- docker + NVIDIA container toolkit
- ffmpeg/ffprobe
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- HuggingFace token for gated models; NGC API key if serving images come from nvcr.io

### Steps

Get the source onto the host (git clone or rsync), then:

```bash
cd ~/aisee
uv sync                        # .venv from uv.lock, aisee installed editable

./aisee install                # checks docker/nvidia/ffmpeg, creates ~/.aisee
./aisee creds set HF_TOKEN     # hidden prompt; lands in ~/.aisee/credentials.json
./aisee creds set NGC_API_KEY  # only if you pull nvcr.io images
```

`./aisee` at the repo root runs the venv'd CLI from anywhere, and bootstraps `.venv` itself on
first use, so the explicit `uv sync` is optional. Symlink it into `~/.local/bin` if you want it
on PATH.

`./aisee uninstall` removes everything AISee put on the host; `--keep-cache` preserves the
downloaded weights.

## Quick Start

```bash
./aisee model install qwen3-vl-30b-a3b-instruct
./aisee assert shot.png -e "the Start button is visible and enabled"
```

The first query starts the API daemon and the model container by itself. Weights download on
first load - the task sits in `model_loading` while that happens, which can be a while for the
big models. The first installed model becomes the default.

More examples:

```bash
./aisee look shot.png -q "What error message is shown?"
./aisee look page.png -q "Where is the search box?" --model holo1-5-7b
./aisee assert run.mp4 -e "the app launches into the main menu" --native
./aisee watch run.mp4 -e "the frame counter increases monotonically" --fps 8
./aisee watch run.mp4 -q "describe what the user does" --fps 2
./aisee status
```

Video can be sent as sampled frames (`--frames N` spread evenly, or `--fps R`) or as the video
itself (`--native`, for models that support it). `--context "..."` / `--context-file f.txt`
passes background the model can't see in the pixels ("the left panel is the scene tree").
`--no-wait` prints the task id and returns; `aisee task show <id>` polls it later.

## Installing Models

```bash
./aisee model install <catalog-slug or HF-id> [--gpu-frac F --image I --port P --idle-timeout S --arg X]
./aisee model list | start <slug> | stop <slug> | logs <slug> | default <slug> | remove <slug>
```

`install` only writes a registry entry (`~/.aisee/models/<slug>.toml`) and picks a port; the
download happens on first start. `stop` frees the GPU but keeps weights and config. `remove`
drops the entry; weights stay in the shared cache.

### Built-In Catalog

The built-in catalog covers seven models measured on a DGX Spark GB10 (2026-07). Installing by
slug applies the serving flags each one needs:

| Slug | GPU slice | Notes |
|---|---|---|
| `qwen3-vl-30b-a3b-instruct` | 0.55 | good default: 32B-class answers at ~5 s (MoE, ~3B active), solid OCR, native video |
| `qwen3-vl-32b-instruct` | 0.70 | deepest synthesis, but 24-45 s per assert on bandwidth-bound GPUs |
| `nvidia-nemotron-nano-12b-v2-vl-nvfp4-qad` | 0.22 | fastest and smallest (NVFP4, ~11 GB); slips digits in dense numbers |
| `holo1-5-7b` | 0.20 | UI element grounding; stills only |
| `cosmos-reason2-8b` | 0.22 | temporal / physical video reasoning |
| `cosmos3-nano` | 0.55 | video reasoning with correct OCR; ~9 min cold load; aarch64 omni image |
| `ui-tars-1-5-7b` | 0.25 | GUI-agent model (action generation later); stills only |

### Other Models

Beyond the catalog, any model works that the serving image's **vLLM can run as a multimodal
chat model** - i.e. it accepts OpenAI-style `chat/completions` with `image_url` content parts.
In practice that is the Qwen-VL family and its derivatives (Holo, UI-TARS), InternVL, Pixtral,
Gemma 3, LLaVA-style models, NVIDIA's Cosmos/Nemotron VL models, and most other open VLMs -
see the [vLLM supported-models list](https://docs.vllm.ai/en/latest/models/supported_models.html)
for the image in use.

Install by HF id and pass whatever serving flags the model needs:

```bash
./aisee model install org/Model --gpu-frac 0.3 --arg --enforce-eager --arg --trust-remote-code
```

Things to know when going off-catalog:

- **Video**: `--native` and `watch` need vLLM `video_url` support for that architecture
  (Qwen-VL, Cosmos, Nemotron-VL have it; grounding-tuned 7Bs like Holo/UI-TARS read a clip as a
  single frame). Frame-based queries (`--frames`/`--fps`) work with any image-capable model.
- **Reasoning models** (answers arrive in `reasoning_content`): handled automatically; give them
  `--max-tokens 2048`.
- **Quantized checkpoints** (NVFP4/FP8/AWQ): vLLM auto-detects the quantization from the
  checkpoint - don't pass `--quantization`.
- **Custom code models**: add `--arg --trust-remote-code`.
- **Different serving image**: `--image` swaps the container image per model (e.g. an
  architecture only supported by a newer vLLM or a vendor build); nvcr.io images need the
  NGC key.
- Per-request caps default to 8 images and 1 video (16 server-sampled frames); AISee's frame
  sampling respects them.

Several models can be installed at once; the running ones' `gpu_frac` slices have to fit in GPU
memory together. Each model runs one inference at a time; tasks queue FIFO per model.

A model idle longer than its `idle_timeout` (default 900 s, `0` disables) is stopped
automatically to free the GPU. The next query targeting it starts it again; the task reports
`model_loading` in the meantime.

## REST API

### Server

```bash
./aisee api start [--port N] [--host 0.0.0.0|127.0.0.1]
./aisee api stop | status
```

`--port` and `--host` persist to `~/.aisee/config.toml`. `0.0.0.0` (the default) serves the LAN,
`127.0.0.1` is local-only. The CLI starts the daemon on demand; `--no-autostart` disables that.

The CLI also works from other machines - `--server http://HOST:PORT` or
`export AISEE_SERVER=http://HOST:PORT` - media files are uploaded with the request.

### Reference

Everything is under `/v1`; OpenAPI schema at `/openapi.json`.

| Method + path | Purpose |
|---|---|
| `GET /v1/describe` | self-description written for LLM consumers: endpoints with examples, task lifecycle, installed models with strengths/weaknesses/pitfalls. Markdown, `?format=json` for structured |
| `GET /v1/health` | liveness + model states |
| `GET /v1/models` | registry with state, port, idle_timeout, last_used, default |
| `POST /v1/models/{slug}/start`, `/stop` | lifecycle (non-blocking) |
| `POST /v1/tasks` | submit, returns `{id}`. Multipart: `files` + `params` (JSON string); or plain JSON with `media_paths` for files already on the host |
| `GET /v1/tasks` | list, filters `?status=` `?model=` |
| `GET /v1/tasks/{id}` | status, progress, timings, result |
| `DELETE /v1/tasks/{id}` | cancel |

```bash
curl -s -X POST http://HOST:PORT/v1/tasks \
  -F files=@screenshot.png \
  -F 'params={"kind":"assert","expectation":"the Start button is visible"}'
# {"id":"3f2a..."}
curl -s http://HOST:PORT/v1/tasks/3f2a...
```

A task moves through `queued`, `preparing_media`, `model_loading` (only when cold), `running`,
and ends `done`, `failed`, or `canceled`. `timings` breaks out `model_load_s`, `media_prep_s`,
`inference_s`.

Auth is off by default. To require a bearer token: `./aisee creds set AISEE_API_TOKEN`, restart
the daemon, and every endpoint except `/v1/describe` and `/v1/health` wants
`Authorization: Bearer <token>`. The CLI picks the token up from the creds store on its own.
Unset the credential and restart to go open again.

## Credentials

There is no credentials file in the repo. Lookup order: environment variable, CLI parameter,
`~/.aisee/credentials.json`, interactive prompt (hidden input, offers to save). Manage the store
with `./aisee creds set|unset|list`; `list` masks values. Keys used: `HF_TOKEN`, `NGC_API_KEY`,
`AISEE_API_TOKEN`. The server never prompts - a task that needs a missing credential fails with
a hint instead.

## Local Data

Everything AISee puts on the host, by who creates it:

**`uv sync` (or the launcher's first run)** creates `.venv/` inside the source checkout - the
Python environment with aisee and its dependencies (fastapi, uvicorn, httpx, pydantic,
python-multipart). Nothing outside the checkout.

**`./aisee install`** creates the state directory `~/.aisee/` (override with `AISEE_HOME`):

```
~/.aisee/
  config.toml          # api host/port, defaults (fps, idle_timeout, task retention)
  credentials.json     # HF/NGC/API tokens, 0600; written by `creds set` or prompts
  models/<slug>.toml   # per-model serving config: image, port, gpu_frac, vllm args
  hf-cache/            # shared model-weights cache, mounted into every container;
                       #   by far the biggest item (tens of GB per model)
  tasks/tasks.db       # sqlite task store (statuses, progress, timings, results)
  tasks/media/<id>/    # uploaded media + derived frames/chunks per task; GC'd with
                       #   the task after the retention period (default 7 days)
  logs/api.log         # API daemon log
  run/api.pid          # daemon pidfile
```

It installs nothing system-wide: docker, the NVIDIA toolkit, and ffmpeg are prerequisites it
only checks for.

**`./aisee model start` (or the first query)** pulls the model's serving image into the docker
image store (`nvcr.io/nvidia/vllm` is ~20 GB, shared by most catalog models) and runs one
container per model, named `aisee-<slug>`. Weights download into `hf-cache/` on the first load.

**`./aisee uninstall`** stops and removes all `aisee-*` containers and deletes `~/.aisee/`
(`--keep-cache` spares the weights). It does not touch the docker images, the source checkout,
or `.venv` - remove those by hand (`docker rmi ...`, `rm -rf ~/aisee`) if you want a clean host.

## Troubleshooting

- Task sits in `model_loading`: normal on first use (weights download, tens of GB) and after an
  idle unload (about 2 min to reload). `./aisee model logs <slug>` tails the vLLM log.
- `failed` with "Free memory ... is less than desired": something else holds the GPU. Stop it
  (`./aisee model stop <slug>`, or whatever non-AISee container is running) and retry.
- HF 403 on a gated model: the token's account must accept the license on the model page first.
- Reasoning models (cosmos family) answering empty or unparseable: handled by a fallback, but
  give them room - `--max-tokens 2048`.
- "Could not open video stream" server-side: shouldn't happen through AISee (video is re-encoded
  to MJPEG-AVI exactly because serving containers often lack H.264), but raw H.264 sent by hand
  will do this.
- After updating the source: `uv sync`, then `./aisee api stop && ./aisee api start`. A running
  daemon keeps executing the old code.

## License

[Apache 2.0](LICENSE)
