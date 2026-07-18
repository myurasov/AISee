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
  - [Authentication](#authentication)
- [MCP Server](#mcp-server)
- [Agent Files](#agent-files)
- [Credentials](#credentials)
- [Local Data](#local-data)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## What Is AISee?

AISee is a tool that gives AI agents eyes. It serves vision-language models in docker containers
on a GPU host and answers questions about images and video files, over a CLI, a REST API, or an
MCP server.

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

- Linux GPU host with an NVIDIA GPU
- [docker](https://docs.docker.com/engine/install/) +
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [ffmpeg](https://ffmpeg.org/download.html) (includes ffprobe; `apt install ffmpeg`)
- [Python](https://www.python.org/downloads/) 3.12+, [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [HuggingFace token](https://huggingface.co/settings/tokens) for gated models
- [NGC API key](https://org.ngc.nvidia.com/setup/api-keys) if serving images come from nvcr.io

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

| Slug | Context | Notes |
|---|---|---|
| `qwen3-vl-30b-a3b-instruct` | 128k | good default: 32B-class answers at ~5 s (MoE, ~3B active), solid OCR, native video |
| `qwen3-vl-32b-instruct` | 64k | deepest synthesis, but 24-45 s per assert on bandwidth-bound GPUs |
| `nvidia-nemotron-nano-12b-v2-vl-nvfp4-qad` | 128k | fastest and smallest (NVFP4, ~11 GB); slips digits in dense numbers |
| `holo1-5-7b` | 128k | UI element grounding; stills only |
| `cosmos-reason2-8b` | 128k | temporal / physical video reasoning |
| `cosmos3-nano` | 64k | video reasoning with correct OCR; ~9 min cold load; omni serving image |
| `cosmos3-super` | 64k/128k | the 64B omnimodel's 32B Reasoner tower only (no generation); 128k on GB10, 64k on 96 GB; ~130 GB first download; needs a vLLM >= 0.24 image |
| `ui-tars-1-5-7b` | 128k | GUI-agent model (action generation later); stills only |

Serving defaults assume the **main mode of operation: a single resident model per GPU**, and
are computed from the detected GPU at `model install` time: `gpu_frac` is **1.0 on discrete
GPUs** and **0.90 on unified-memory systems** (GB10/Grace class, where the GPU pool is also
system RAM), and the context window is the largest standard size (up to 128k) whose KV cache
fits next to the model's weights. On the known tiers: **GB10** (~120 GiB unified) serves the
whole catalog at 128k; a **96 GB** discrete card serves everything at 128k except the dense 32B
(64k - its 128k KV cache alone is ~34 GiB); a **48 GB** card fits the 7-17 GiB models at 128k
and Cosmos3-Nano at 32k, while the two big Qwens (~62 GiB weights) do not fit at all (install
warns). Media budgets are 16 images / 24 video frames per request (24 keeps each frame at ~720p - the video pixel budget is shared across frames). Execution mode is also per-GPU:
unified-memory systems serve with `--enforce-eager` (CUDA graphs measured slower there),
discrete GPUs keep CUDA graphs (3-4x faster). Each model runs up to `concurrency` inferences
in parallel (default 3; vLLM batches them) - concurrent bursts gain ~1.4-2x and `watch`
chunks are processed in parallel. Context length is the
expensive knob - vLLM reserves KV-cache memory for the full `max_model_len` inside the model's
`gpu_frac` slice, so raising it costs GPU memory even for short requests; override with
`--max-model-len` / `--gpu-frac` at install.

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
- Per-request budgets default to 16 images and 1 video (24 server-sampled frames, keeping
  each frame at ~720p); AISee's frame sampling respects them. There is no hard video-length
  limit - only temporal resolution (the frame budget spread over the clip); use `watch` for
  long videos.

Several models can be installed at once, but with the single-model defaults only one fits the
GPU at a time - to co-locate models, lower `gpu_frac` and `max_model_len` per model so the
slices sum under 1.0 (weights + KV must fit each slice; e.g. an 8B at 0.25/32k next to the MoE
at 0.55/32k). A start that would push the running models' `gpu_frac` sum over 1.0 is refused
up front with a clear error (HTTP 409) instead of letting the container crash-loop. Tasks
queue FIFO per model, with up to `concurrency` running at once.

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

| Method + path | Auth | Purpose |
|---|---|---|
| `GET /v1/describe` | open | self-description written for LLM consumers: endpoints with examples, task lifecycle, installed models with strengths/weaknesses/pitfalls. Markdown, `?format=json` for structured, `?flavor=mcp` for the MCP tool guide |
| `GET /v1/health` | open | liveness + model states |
| `GET /v1/gpu` | consumer | live GPU utilization/memory/power/temperature |
| `GET /v1/models` | consumer | registry with state, port, idle_timeout, last_used, default |
| `GET /v1/catalog` | consumer | built-in catalog with installed flags |
| `GET /v1/config` | consumer | effective global configuration (api + defaults) |
| `POST /v1/tasks` | consumer | submit, returns `{id}`. Multipart: `files` + `params` (JSON string); or plain JSON with `media_paths` for files already on the host |
| `GET /v1/tasks` | consumer | list, filters `?status=` `?model=` |
| `GET /v1/tasks/{id}` | consumer | status, progress, timings, result |
| `DELETE /v1/tasks/{id}` | consumer | cancel |
| `GET /v1/tasks/{id}/media` | consumer | media facts per file: kind, dimensions, duration, frames, size |
| `GET /v1/tasks/{id}/media/{i}` | consumer | download the task's i-th media file (`/thumb` for a JPEG thumbnail) |
| `GET /v1/blobs/{sha256}` | consumer | upload-dedup probe: `{exists, size}` |
| `POST /v1/blobs` | consumer | upload media into the content store without a task |
| `POST /v1/models` | admin | install (`{"name": <catalog slug or HF id>, ...overrides}`) |
| `DELETE /v1/models/{slug}` | admin | uninstall (weights stay cached) |
| `POST /v1/models/{slug}/start`, `/stop` | admin | lifecycle (non-blocking) |

```bash
curl -s -X POST http://HOST:PORT/v1/tasks \
  -F files=@screenshot.png \
  -F 'params={"kind":"assert","expectation":"the Start button is visible"}'
# {"id":"3f2a..."}
curl -s http://HOST:PORT/v1/tasks/3f2a...
```

A task moves through `queued`, `preparing_media`, `model_loading` (only when cold), `running`,
and ends `done`, `failed`, or `canceled`. `timings` breaks out `model_load_s`, `media_prep_s`,
`inference_s`, and (once finished) `total_s` - wall-clock from submission to the terminal
state, also shown in the console's tasks table.

Uploads are deduplicated: media is stored content-addressed by the SHA-256 of the file bytes
(kept for `blob_ttl_hours`, default 24 h; reuse refreshes it), and a media entry can be
`"sha256:<hash>"` instead of a file - probe with `GET /v1/blobs/{sha256}` first (hash via
`sha256sum` / `shasum -a 256` / python `hashlib`). The CLI and the web console negotiate this
automatically, so re-submitting the same video skips the upload; `POST /v1/blobs` uploads
media without creating a task (this is also how remote MCP clients get local files to the
server).

### Authentication

Auth is off by default. Two optional bearer tokens split access into roles:

- `AISEE_API_TOKEN` (**consumer**): guards the query/read endpoints - submitting and reading
  tasks, listing models/catalog/GPU stats. Hand this one to agents and users of the service.
- `AISEE_ADMIN_TOKEN` (**admin**): guards model management (install/uninstall/start/stop).
  Accepted on consumer endpoints too, so an admin needs only one token.

With only the consumer token set, it guards everything (single-token mode). With both set, a
consumer token on an admin endpoint gets `403`; a missing or wrong token gets `401`. The
console (`/`), `/v1/describe`, and `/v1/health` are always open.

```bash
./aisee creds set AISEE_API_TOKEN     # consumer token
./aisee creds set AISEE_ADMIN_TOKEN   # admin token
```

Tokens set through `creds set` apply immediately (the store is read per request); tokens set
as environment variables of the daemon require a restart. The CLI picks tokens up from env or
the creds store on its own (admin preferred when present; `--token` overrides). Unset the
credentials to go open again.

## MCP Server

The API server also speaks MCP (Model Context Protocol, streamable HTTP) at
`http://HOST:PORT/mcp` - nothing to install on the client, point an agent harness (Claude
Code, Cursor, etc.) at the URL. It exposes AISee as native tools: `look`, `assert_visual`,
`watch`, `list_models`, `list_tasks`, `get_task`, `cancel_task`, `describe`, `health`. It is
a thin adapter over the same REST API and intentionally carries **consumer capabilities
only** - model management is not reachable over MCP.

Register it in an MCP client config:

```json
{
  "mcpServers": {
    "aisee": {
      "type": "http",
      "url": "http://HOST:PORT/mcp",
      "headers": { "Authorization": "Bearer <consumer token>" }
    }
  }
}
```

The `headers` entry is only needed when auth is enabled; `/mcp` is guarded like any consumer
endpoint. Query tools block until the result is ready (a cold model can take minutes);
`watch` accepts `wait=false` to return a task id for polling with `get_task`. Media paths
are resolved **on the AISee host** - the files must already exist there (transfer them
first, or use the REST API, which uploads).

## Agent Files

Two ready-made role instructions for AI agents live at the repo root:

- [`aisee.consumer.agent.md`](aisee.consumer.agent.md) - for agents that *use* AISee to see:
  query kinds, CLI/REST/MCP access, behavior to plan around, limitations.
- [`aisee.admin.agent.md`](aisee.admin.agent.md) - for agents that *operate* an AISee host:
  installing AISee locally or on a remote machine, tokens, model and server management,
  troubleshooting.

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
python-multipart, mcp). Nothing outside the checkout.

**`./aisee install`** creates the state directory `~/.aisee/` (override with `AISEE_HOME`):

```
~/.aisee/
  config.toml          # api host/port, defaults (fps, idle_timeout, task retention, blob TTL)
  credentials.json     # HF/NGC/API tokens, 0600; written by `creds set` or prompts
  models/<slug>.toml   # per-model serving config: image, port, gpu_frac, vllm args
  hf-cache/            # shared model-weights cache, mounted into every container;
                       #   by far the biggest item (tens of GB per model)
  tasks/tasks.db       # sqlite task store (statuses, progress, timings, results)
  tasks/blobs/         # content-addressed uploads (sha256-named; upload dedup); GC'd
                       #   after blob_ttl_hours (default 24), refreshed on reuse
  tasks/media/<id>/    # per-task media (hardlinks into blobs/) + derived frames/chunks;
                       #   GC'd with the task after task_ttl_hours (default 24)
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
- `failed to discover GPU vendor from CDI` (docker cannot see the GPU): the NVIDIA Container
  Toolkit is missing or unconfigured - install it, then `sudo nvidia-ctk runtime configure
  --runtime=docker && sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml && sudo
  systemctl restart docker`. `./aisee install` checks for this.
- Reasoning models (cosmos family) answering empty or unparseable: handled by a fallback, but
  give them room - `--max-tokens 2048`.
- "Could not open video stream" server-side: shouldn't happen through AISee (video is re-encoded
  to MJPEG-AVI exactly because serving containers often lack H.264), but raw H.264 sent by hand
  will do this.
- After updating the source: `uv sync`, then `./aisee api stop && ./aisee api start`. A running
  daemon keeps executing the old code.

## License

[Apache 2.0](LICENSE)
