---
name: aisee-admin
description: >
  Operate an AISee host: install AISee on a (possibly remote) GPU machine, manage models
  (install/uninstall/start/stop), run the API server, configure the consumer/admin auth
  tokens, and troubleshoot serving. Adopt this agent to set up or administer AISee; for
  querying it (look/assert/watch), see aisee.consumer.agent.md.
triggers: ["install aisee", "set up aisee", "aisee admin", "manage aisee models",
           "aisee server", "aisee tokens"]
---

# AISee admin agent (for AI agents that operate an AISee host)

AISee is a tool that gives AI agents eyes and ears; this role runs the host that provides them. You
install AISee, manage models and the API daemon, and hand consumers a URL plus (optionally) a
consumer token. Admin actions are the modifying ones: model install/uninstall/start/stop.
Everything a consumer can do, you can do too - the admin token is accepted everywhere.

## Installing AISee on a host (local or remote)

Prerequisites on the GPU host: Linux, NVIDIA GPU + driver, docker + NVIDIA Container Toolkit,
ffmpeg, Python 3.12+, uv. Convention: source at `~/aisee`, venv at `~/aisee/.venv`, all state
under `~/.aisee/`.

Local (a shell on the host):

```bash
git clone https://github.com/myurasov/AISee ~/aisee && cd ~/aisee
uv sync                          # or skip: ./aisee bootstraps its own .venv
./aisee install                  # verifies docker/GPU/ffmpeg/toolkit, creates ~/.aisee
./aisee creds set HF_TOKEN       # gated models (the HF account must accept model licenses)
./aisee creds set NGC_API_KEY    # only if serving images come from nvcr.io
```

Remote (you have ssh access to the host): run exactly the same commands over ssh. Either
clone on the host, or push a local working copy:

```bash
ssh HOST 'git clone https://github.com/myurasov/AISee ~/aisee'
# or, from a local checkout (deploys uncommitted changes too):
rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ \
      --exclude '*.egg-info' ./ HOST:aisee/
ssh HOST 'cd ~/aisee && uv sync && ./aisee install'
ssh HOST '~/aisee/aisee creds set HF_TOKEN <token>'
```

`./aisee install` reports anything missing (docker daemon, NVIDIA Container Toolkit/CDI,
ffmpeg) with the exact fix commands. Re-run it until it prints `install: ok`.

After updating the source on a host: `uv sync`, then restart the API - `./aisee api stop &&
./aisee api start`, or `systemctl restart aisee-api` on a persistent (systemd) install, where
the stop/start pair silently no-ops - a running daemon keeps executing old code. `res/*`
(console, describe template) is read per request and needs no restart. When a release
changes the default serving image (1.1: `nvcr.io/nvidia/vllm:26.08-py3`), pull it first
(`docker pull <image>`, ~20+ GB) - otherwise the first start of each model pulls it while
holding the one-cold-load-at-a-time lock - and stop running models afterwards so they come
back on it (see Managing models).

## Auth: consumer and admin tokens

Two optional bearer tokens, stored as ordinary credentials (env var > `~/.aisee/credentials.json`):

- `AISEE_API_TOKEN` (**consumer**): when set, guards the query/read endpoints - submitting
  and reading tasks, listing models/catalog/GPU stats. Give this one to consumers.
- `AISEE_ADMIN_TOKEN` (**admin**): when set, guards the management endpoints - model
  install/uninstall/start/stop. Accepted on consumer endpoints too. Keep it private.

Semantics: with only `AISEE_API_TOKEN` set, that single token guards everything (legacy
single-token mode). With both set, a consumer token on an admin endpoint gets **403**; a
missing/wrong token gets **401**. `/`, `/v1/describe`, and `/v1/health` are always open.

```bash
./aisee creds set AISEE_API_TOKEN     # consumer
./aisee creds set AISEE_ADMIN_TOKEN   # admin
```

Tokens set with `creds set` apply immediately (the store is read per request); tokens set as
env vars of the daemon need a restart.

The CLI picks tokens up automatically (admin preferred when present; `--token` overrides).
The MCP endpoint (`/mcp` on the API server, streamable HTTP) is guarded by the consumer
token and carries consumer capabilities only - it cannot manage models by design.

## Running the API server

```bash
./aisee api start [--port N] [--host 0.0.0.0|127.0.0.1]   # persisted to ~/.aisee/config.toml
./aisee api stop | status
```

`0.0.0.0` (default) serves the LAN; the daemon must run on the GPU host itself. Log:
`~/.aisee/logs/api.log`. A single-file web console at `/` covers status, queries, tasks,
models (admin actions need the admin token, entered on its Server tab), and live GPU stats.

For an installation that survives host reboots, run the server under systemd instead of the
one-off daemon: see "Persistent installation" in README.md for the unit. Prefer the
**system** unit (`User=<user>`, `ExecStart=<checkout>/.venv/bin/python -P -m aisee.server`,
`After=docker.service`); a user unit + linger also works but inherits the user manager's
frozen group list - if the `docker` group was added after that manager started, the API
cannot see containers (all models report `installed` while their containers run) until the
manager restarts. Model containers need no unit; they auto-restart via docker's
`unless-stopped` policy. Under systemd, restart with `systemctl restart aisee-api`
(`./aisee api stop && ./aisee api start` silently no-ops there: stop finds no pidfile,
start sees the healthy API); logs via `journalctl -u aisee-api`.

## Managing models

```bash
./aisee model install <catalog-slug or HF-id> [--gpu-frac F --max-model-len N --image I ...]
./aisee model list | start <slug> | stop <slug> | logs <slug> | default <slug> | remove <slug>
```

- `install` writes a registry entry (`~/.aisee/models/<slug>.toml`) and picks a port; weights
  (tens of GB) download on first start - the task/model sits in `model_loading` meanwhile.
  Serving settings are auto-sized for the detected GPU from the model's absolute GiB
  requirement (mem_gib -> serving fraction; context = largest size up to the checkpoint's
  native limit whose KV cache fits the slice; media budgets; eager vs CUDA graphs);
  install warns when the weights cannot fit.
- The first installed model becomes the default. Models co-reside when their GiB
  requirements fit together (the ~11 GiB audio pair next to a VLM is the normal case);
  to co-locate more, lower `--gpu-frac`/`--max-model-len` per model.
- Idle models auto-stop after `idle_timeout` (default 3600 s; `0` never) and restart on the
  next query. `model stop` also cancels a start you issued that is still in progress. A stop
  does not stick while tasks for that model are queued or running - the queue needs the
  model and starts it again; cancel those tasks first.
- Serving image: catalog models on the NGC default image follow each release's default
  (currently `nvcr.io/nvidia/vllm:26.08-py3`), including entries written by older
  releases, and so do the entries retired in 1.1.0b1 (the Qwen3-VL trio and Nemotron-Nano);
  the Cosmos3 models (vllm-omni images) and audio models keep their own images, and
  off-catalog installs - including models dropped from the catalog outright, like Holo1.5
  and UI-TARS-1.5 - keep the image they were installed with (reinstall to move one). A
  container that was already running when you upgraded keeps its old image until it is
  recreated - `model list` shows `(still on <image>)`; `model stop <slug>` switches it on
  the next query (models with `idle_timeout = 0` never switch by themselves).
- Pinning / rollback: `--image I` at install pins a model to I (`(pinned)` in the install
  output). A reinstall recomputes every setting and drops hand edits (e.g. a tuned
  `max_images`), and a reinstall without `--image` unpins. To pin or roll back WITHOUT
  losing tuning, edit the model's TOML instead: set
  `image = "nvcr.io/nvidia/vllm:26.06-py3"` and add `image_pinned = true`, then
  `model stop <slug>`. (Before 1.1.0a2 there was no pin: an explicit 26.06 was treated as
  the old default and moved to 26.08 - redo such rollbacks on 1.1.0a2+.)
- Off-catalog Qwen3-VL derivatives: add `video_loader = "opencv"` to the model's TOML
  (then `model stop`) to get the catalog behavior - up to `video_frames` frames spread
  evenly over the clip, and every frame of a shorter clip or watch chunk. Without it, vLLM
  >= 0.24 samples their native video at a fixed 2 fps and ignores the frame cap, and any
  vLLM resamples a clip with no more frames than the cap to 2 fps.
- Kernel caches: models on NGC and vllm-omni images keep compiled Triton kernels,
  FlashInfer autotune picks, and the CUDA driver's PTX JIT cache in
  `~/.aisee/cache/jit/<image>-<id>/` (one directory per image build, root-owned, typically
  a few hundred MB; the CUDA part is capped at 1 GiB). On a GB10 they cut Nemotron's engine
  init from ~207 s to ~44 s and its first request after a load from ~140 s to seconds
  (Cosmos3-Nano's from ~24 s to ~9 s); on an RTX PRO 6000, Cosmos3-Nano's first request
  went from ~10 s to ~1.5 s - only the first load on a new image pays.
  Picks measured while the GPU was busy persist too; `sudo rm -rf ~/.aisee/cache/jit` resets
  everything (the next start re-tunes). Directories of image builds that are no longer on the
  host (`docker rmi`, or a tag re-pulled to a new build) are pruned when the API starts - the
  API log names them.
- Concurrency: a model TOML's `concurrency` (default 3) applies to the task queue live, but
  the container's vLLM sequence cap (`--max-num-seqs`: concurrency squared, at least 16 and
  at most 256, unless `extra_args` sets it) is fixed when the container starts - after
  raising `concurrency` above 4, `model stop <slug>` so the next start sizes it.
- Watch rate: config.toml `[defaults] fps` is the default watch sampling rate - 2 since
  1.1.0a3. A config.toml from before 1.1.0a3 still carries the old default `fps = 3.0`
  (every config write stored all defaults): the first load after the upgrade rewrites it
  to 2 once and stamps `[meta] config_version = 2`; an fps you set after that sticks.
  Watch now delivers the configured rate; before, the Qwen3-VL/Cosmos family got at most
  2 fps whatever was set, and at 3 fps its free-form chunk narration degraded in testing.
  A model TOML's own `fps` overrides the default. A Cosmos3 container that was running
  when you upgraded keeps the old 2 fps loader until it is recreated - `model stop <slug>`.
- Thinking toggle (the Qwen3.5-family entries, `thinking_toggle = true`): config.toml
  `[defaults] thinking` (default false since 1.1.0b1) decides what a call without
  `thinking` gets; the per-call flag wins. A config.toml from an older version that carried
  the old default `thinking = true` is switched to false once, on the first load after the
  upgrade (`[meta] config_version = 3`); set it back to true afterwards and it sticks.
  Thinking calls sample at temperature 0.6 with an 8192-token budget. A model TOML's
  `thinking_sampling = { temperature = 1.0, top_p = 0.95 }` table overrides that sampling key
  by key for one model (`max_tokens`, `messages`, `stream` and other request fields are
  ignored; `enable_thinking` is always kept). These models are hybrids: vLLM serves them
  without prefix caching, so a repeat question about the same media re-reads it in full. To
  try it, add `"--enable-prefix-caching"` to the `extra_args` list in the model's TOML, then
  `model stop <slug>` (`model install --arg` would replace the catalog's flags, not add to
  them). Models installed before their catalog entry existed (by HF id on 1.1.0a3) keep the
  flags they were installed with - reinstall them by slug to get the catalog's.
- Remote equivalents exist over REST with the admin token:
  `POST /v1/models {"name": ...}`, `DELETE /v1/models/{slug}`,
  `POST /v1/models/{slug}/start|stop` - so a remote admin does not need ssh once the API
  is up. `GET /v1/catalog` lists the built-in catalog with installed flags.

Consult `GET /v1/describe` (or `README.md`) for the catalog with per-model
strengths/weaknesses/pitfalls and the current serving configuration.

Audio models (`parakeet-tdt-0.6b-v3` for transcribe, `pyannote/speaker-diarization-3.1` for
diarize) share the exact same lifecycle. Differences to know: their serving images are built
locally on the host from `res/serving/` on first start (~10-20 min one-time; progress shows
"building serving image"); their containers GPU-gate at startup and exit nonzero when
inference is not on CUDA (read `model logs <slug>` for the FATAL line - never serve a silent
CPU fallback); pyannote weights are HF-gated - the HF_TOKEN account must accept the license
forms on speaker-diarization-3.1, segmentation-3.0, AND wespeaker-voxceleb-resnet34-LM. The
first model installed per capability becomes `defaults.default_transcribe_model` /
`default_diarize_model`. Audio containers run with a hard RAM cap (mem_limit in the model
TOML; ASR 40g - long-form timestamp alignment peaks ~16 GB on a 79-min file) and a high
oom-score-adj, so under memory pressure THEY die (task fails cleanly, container restarts)
rather than the host or the VLM. On unified-memory hosts do not schedule hour-scale
transcriptions while a big vision model is resident - stop it or let it idle-unload first.

## Sizing models per GPU (context, image budgets, memory)

`install` auto-sizes everything, so normally you do nothing. This section is for when you
want a DIFFERENT configuration (a bigger context on a small card, co-residency headroom,
a larger image batch) or need to predict what a GPU will give before buying/renting one.

**The sizing model.** Every catalog entry carries measured, GPU-independent components;
a host configuration follows from them:

```
grant   = min(catalog mem_gib, cap x VRAM)     # cap: 0.97 discrete, 0.92 unified
KV pool = grant - weights_gib - 4 (runtime)
context = largest candidate (native, 128k, 64k, 32k, 16k, 8k) whose KV cost fits:
          kv_cost(c) = kv_gib_128k x (c / 128k) x (0.5 if fp8 KV)
max_images = clamp((context - 8192) / tokens_per_image, 4..120)
```

Measured components (fp8 KV is on by default for the Qwen and Cosmos families):

| model | weights | KV GiB/128k | fp8 | tok/img | native ctx |
|---|---|---|---|---|---|
| qwen3-6-35b-a3b | 67 | 2.5 | yes | 2200 | 256k |
| qwen3-8-27b | 52 | 8 | yes | 2200 | 256k |
| qwen3-5-9b | 18 | 4.5 (incl. MTP) | yes | 2200 | 256k |
| cosmos-reason2-8b | 17 | 17.5 | yes | 2200 | 256k |
| cosmos3-nano | 32 | 14.5 | yes | 2200 | 256k |
| cosmos3-super | 64 | 30.5 | yes | 2200 | 256k |

**What the auto-sizer produces on common GPUs** (context / image budget; "-" = weights
do not fit). Columns are VRAM tiers; map cards to tiers: T4 / V100-16G / RTX 4080 -> 16, RTX 3090 / 4090 / L4 -> 24,
RTX 5090 -> 32, A100-40G -> 40, L40S / RTX 6000 Ada / A6000 -> 48, A100-80G / H100 -> 80,
RTX PRO 6000 Blackwell -> 96, DGX Spark GB10 (unified) -> 120u, H200 -> 141:

| model | 16 | 24 | 32 | 40 | 48 | 80 | 96 | 120u | 141 |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-6-35b-a3b | - | - | - | - | - | 256k/115 | 256k/115 | 256k/115 | 256k/115 |
| qwen3-8-27b | - | - | - | - | - | 256k/115 | 256k/115 | 256k/115 | 256k/115 |
| qwen3-5-9b | - | 64k/26 | 256k/115 | 256k/115 | 256k/115 | 256k/115 | 256k/115 | 256k/115 | 256k/115 |
| cosmos-reason2-8b | - | 32k/11 | 128k/55 | 128k/55 | 256k/115 | 256k/115 | 256k/115 | 256k/115 | 256k/115 |
| cosmos3-nano | - | - | - | 32k/11 | 128k/55 | 256k/115 | 256k/115 | 256k/115 | 256k/115 |
| cosmos3-super | - | - | - | - | - | 64k/26 | 128k/55 | 256k/115 | 256k/115 |

(No catalog vision model fits a 16 GB card. Computed with the exact install logic from the
component table; audio models are not context-sized - parakeet needs ~7 GiB and pyannote
~4 GiB on any GPU. The entries retired in 1.1.0b1 - the Qwen3-VL trio and Nemotron-Nano -
keep the sizing they were installed with; README lists their successors.)

**Changing the configuration.** Reinstall with overrides - reinstalling preserves the
port, idle_timeout, and default flag, and recomputes everything else consistently
(hand-editing `max_model_len` in the TOML does NOT resize the image budget):

```bash
./aisee model install <slug> --max-model-len 65536   # smaller ctx -> smaller KV pool ->
                                                     # frees memory for co-residency
./aisee model install <slug> --gpu-frac 0.5          # hard-cap the memory slice; context
                                                     # auto-shrinks to what still fits
```

- **More images per request**: only a bigger context buys more (the budget is derived);
  on a card stuck at a small context, pick a model with cheaper KV (see table - e.g.
  the hybrid qwen3-5-9b reaches 256k/115img on 32 GB, where cosmos-reason2-8b gets 128k/55).
- **A context the auto-sizer refused**: it does not fit - the KV pool is the only
  flexible part, weights + 4 GiB runtime are fixed. Going bigger means fp8 KV (already
  default where supported), a smaller model, or a bigger GPU.
- **Will it actually START on this host**: admission also requires free memory NOW -
  need + margin (10 GiB unified for >= 30 GiB loads, 3 GiB small, 2 GiB discrete), and
  a serving model consumes ~4 GiB above its grant. On a 120 GiB GB10 that puts the
  practical ceiling at ~102 GiB (hence cosmos3-super's catalog size).
- After any resize, `GET /v1/describe` shows the resulting context + image budget per
  model - verify there, not in the TOML.

## Troubleshooting

- `install` says the NVIDIA Container Toolkit is missing / `failed to discover GPU vendor
  from CDI`: install the toolkit, then `sudo nvidia-ctk runtime configure --runtime=docker &&
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml && sudo systemctl restart docker`.
- Model stuck loading: `./aisee model logs <slug>` - usually a weight download (HF can
  throttle to ~10-15 MB/s; the largest models take tens of minutes on first load).
- HF 403 on a gated model: the token's account must accept the license on the model page.
- Starting a model that does not fit next to the running ones - by stated GiB
  requirements plus a system reserve, or by actually-free GPU memory - is refused up front
  (HTTP 409 / a clear GiB-denominated error) before any container work; stop a resident
  model first or co-locate with smaller slices. On unified-memory hosts a large model is
  also refused while audio jobs are in flight (the cold load would starve them - retry
  after they finish), and a start right after a big job ends can be refused for a minute
  until memory actually settles. "Free memory ... is less than desired" can still
  happen when a non-AISee process holds the GPU - stop it and retry.
- Tasks orphaned in `model_loading` after a daemon crash are requeued automatically at the
  next `api start`.
- Uploaded media is content-addressed (SHA-256 of the bytes) under `~/.aisee/tasks/blobs/`
  so repeat uploads are skipped; blobs age out after `blob_ttl_hours` (config.toml
  `[defaults]`, default 24, 0 disables GC; reuse refreshes the clock), and tasks keep
  hardlinked copies, so blob GC never breaks a task.
  Consumers can hash locally (`sha256sum` / `shasum -a 256`), probe `GET /v1/blobs/{sha}`,
  and pass `sha256:<hash>` media refs - see `aisee.consumer.agent.md`.
- `./aisee uninstall` removes all AISee containers and `~/.aisee` (`--keep-cache` spares the
  downloaded weights and kernel caches); the source checkout and docker images stay.

## What to hand a consumer

1. Server URL: `http://HOST:PORT` (check with `./aisee status`); MCP clients use
   `http://HOST:PORT/mcp`.
2. The consumer token, if auth is on - never the admin token.
3. Pointer to `aisee.consumer.agent.md` (this repo) and `GET /v1/describe` for usage.
