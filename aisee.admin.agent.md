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

AISee is a tool that gives AI agents eyes; this role runs the host that provides them. You
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

After updating the source on a host: `uv sync`, then restart the API (`./aisee api stop &&
./aisee api start`) - a running daemon keeps executing old code. `res/*` (console, describe
template) is read per request and needs no restart.

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

## Managing models

```bash
./aisee model install <catalog-slug or HF-id> [--gpu-frac F --max-model-len N --image I ...]
./aisee model list | start <slug> | stop <slug> | logs <slug> | default <slug> | remove <slug>
```

- `install` writes a registry entry (`~/.aisee/models/<slug>.toml`) and picks a port; weights
  (tens of GB) download on first start - the task/model sits in `model_loading` meanwhile.
  Serving settings (gpu_frac, context, media budgets, eager vs CUDA graphs) are auto-sized
  for the detected GPU; install warns when the weights cannot fit.
- The first installed model becomes the default. Single resident model per GPU is the main
  mode; to co-locate models lower each `--gpu-frac`/`--max-model-len` so slices sum < 1.0.
- Idle models auto-stop after `idle_timeout` (default 900 s) and restart on the next query.
- Remote equivalents exist over REST with the admin token:
  `POST /v1/models {"name": ...}`, `DELETE /v1/models/{slug}`,
  `POST /v1/models/{slug}/start|stop` - so a remote admin does not need ssh once the API
  is up. `GET /v1/catalog` lists the built-in catalog with installed flags.

Consult `GET /v1/describe` (or `README.md`) for the catalog with per-model
strengths/weaknesses/pitfalls and the current serving configuration.

## Troubleshooting

- `install` says the NVIDIA Container Toolkit is missing / `failed to discover GPU vendor
  from CDI`: install the toolkit, then `sudo nvidia-ctk runtime configure --runtime=docker &&
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml && sudo systemctl restart docker`.
- Model stuck loading: `./aisee model logs <slug>` - usually a weight download (HF can
  throttle to ~10-15 MB/s; the largest models take tens of minutes on first load).
- HF 403 on a gated model: the token's account must accept the license on the model page.
- Starting a model whose gpu_frac does not fit next to the running ones is refused up front
  (HTTP 409 / a clear CLI error) before any container work - stop a resident model first or
  co-locate with smaller --gpu-frac slices. "Free memory ... is less than desired" can still
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
  downloaded weights); the source checkout and docker images stay.

## What to hand a consumer

1. Server URL: `http://HOST:PORT` (check with `./aisee status`); MCP clients use
   `http://HOST:PORT/mcp`.
2. The consumer token, if auth is on - never the admin token.
3. Pointer to `aisee.consumer.agent.md` (this repo) and `GET /v1/describe` for usage.
