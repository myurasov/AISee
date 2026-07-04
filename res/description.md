# AISee v0.1.0 — API guide for AI agents

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

1. `POST /v1/tasks` — returns `{"id": "..."}` immediately (non-blocking).
2. Poll `GET /v1/tasks/{id}` every 2-5 s. `status` walks through:
   `queued -> preparing_media -> model_loading (only if the model is cold) -> running -> done`
   (`failed` / `canceled` are terminal too). `progress` holds a human-readable `step` + `detail`,
   and for `watch` a `chunk: {i, n, t_start, t_end}` counter.
3. **`model_loading` can take minutes** (cold model start; the largest models take ~9 minutes on
   first load). This is normal — keep polling; `progress.detail` explains what is happening.
4. Read `result` when `status == "done"`; on `failed`, `error.message` says why.

Task kinds and their `result` shapes:
- `look` — free-form question about the media. Result: `{"answer": "<text>"}`.
- `assert` — pass/fail judgment of an `expectation`. Result:
  `{"pass": bool, "reason": str, "evidence": str}`. Use for visual regression / e2e checks.
- `watch` — chunked whole-video analysis. With `expectation`: per-chunk verdicts +
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

### `cosmos-reason2-8b` — installed
- HF id: `nvidia/Cosmos-Reason2-8B`; native video: yes; license: NVIDIA Open Model
- **Strengths:** Purpose-built temporal / physical video reasoning; fast (~5 s asserts); handles native video well.
- **Weaknesses:** Not a UI specialist; weaker on dense-text stills than the Qwen/Holo family.
- **Pitfalls:** Reasoning model: answers can arrive in reasoning_content with content null (AISee falls back automatically); give it headroom in max_tokens.

### `cosmos3-nano` — installed
- HF id: `nvidia/Cosmos3-Nano`; native video: yes; license: NVIDIA Open Model
- **Strengths:** Strong temporal/physical video reasoning; correct OCR; handles native video.
- **Weaknesses:** Slow to come up; one-time ~59 s first-call warmup after load.
- **Pitfalls:** Serves only on the vllm-omni image (aarch64) with architecture override Cosmos3ForConditionalGeneration; ~9-minute quiet init before weight shards load - it is not hung.

### `holo1-5-7b` — installed
- HF id: `Hcompany/Holo1.5-7B`; native video: no; license: Apache-2.0
- **Strengths:** Pixel-precise UI element grounding (computer-use lineage); very fast stills (~1.4 s OCR, ~2.4 s grounding); low memory (~16 GB).
- **Weaknesses:** Stills-only: reads a video clip as a single frame. Terse answers.
- **Pitfalls:** Hangs during CUDA-graph capture unless served with --enforce-eager.

### `nvidia-nemotron-nano-12b-v2-vl-nvfp4-qad` — installed
- HF id: `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD`; native video: yes; license: NVIDIA Open Model (commercial use permitted)
- **Strengths:** Fastest overall (NVFP4, ~11 GB resident): ~4-7 s stills, ~1 s OCR/grounding; handles native video; smallest GPU footprint.
- **Weaknesses:** Fumbled a dense number in testing (OCR digit slip) - do not trust it for exact figures.
- **Pitfalls:** Needs --trust-remote-code and --enforce-eager. NVFP4 quantization is auto-detected - do NOT pass --quantization.

### `qwen3-vl-30b-a3b-instruct` **(default)** — installed
- HF id: `Qwen/Qwen3-VL-30B-A3B-Instruct`; native video: yes; license: Apache-2.0
- **Strengths:** Recommended default. 32B-class quality at small-model speed (MoE, ~3B active params): ~5-7 s stills, correct OCR on dense numbers, handles native video, fast element grounding (~1.3 s).
- **Weaknesses:** The full ~62 GB of BF16 weights must be resident despite the speed; not a specialist at physical/temporal reasoning.
- **Pitfalls:** Needs --enforce-eager on GB10-class hardware. First install downloads ~62 GB.

### `qwen3-vl-32b-instruct` — installed
- HF id: `Qwen/Qwen3-VL-32B-Instruct`; native video: yes; license: Apache-2.0
- **Strengths:** Deepest synthesis / long narration; correct OCR; handles native video.
- **Weaknesses:** 4-9x slower than small/MoE models on bandwidth-bound GPUs (24-45 s per still assert). Use only when maximum reasoning depth matters.
- **Pitfalls:** gpu_frac below ~0.70 crash-loops ('No available memory for the cache blocks').

### `ui-tars-1-5-7b` — installed
- HF id: `ByteDance-Seed/UI-TARS-1.5-7B`; native video: no; license: Apache-2.0
- **Strengths:** GUI-agent lineage: can emit click/type actions (future action generation); correct OCR; solid still judgments.
- **Weaknesses:** Stills-only: reads a video clip as a single frame.
- **Pitfalls:** Needs --trust-remote-code and --enforce-eager.

## Practical limits

- Image inputs are capped per model (typically 8 per request) — sample videos accordingly.
- Prefer `assert` over `look` when you need a machine-checkable verdict.
- Pass `context` for domain knowledge the model can't see ("the left panel is the scene tree").
- One inference runs at a time per model; tasks queue FIFO. Idle models are auto-stopped after
  their idle timeout and transparently restarted on the next task (expect `model_loading`).
