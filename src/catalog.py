# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Seed model catalog: install by slug without hand-writing serving flags.

Entries carry serving requirements plus agent-facing strengths / weaknesses / pitfalls
(measured on a DGX Spark GB10, 2026-07) that feed the /v1/describe model guide.
"""

DEFAULT_IMAGE = "nvcr.io/nvidia/vllm:26.08-py3"  # vLLM 0.27.1 (26.06 was 0.22.1)
# earlier defaults, still recorded in registry TOMLs on hosts installed before the bump;
# the registry maps them to DEFAULT_IMAGE at read time so an upgrade reaches installed models
LEGACY_DEFAULT_IMAGES = ("nvcr.io/nvidia/vllm:26.06-py3",)

# Serving requirements are stated in absolute GiB (mem_gib) and adapted to the detected
# GPU at install time (registry.gpu_profile / fit_max_model_len): the serving fraction is
# mem_gib / GPU memory (clamped to the host cap), and max_model_len is the largest
# context (up to ctx_native) whose KV cache fits next to the weights inside that slice
# (entries carry measured weights_gib / kv_gib_128k; an fp8 KV cache halves the KV cost).
# Known tiers: GB10 (~120 GiB unified), 96 GB and 48 GB discrete.
DEFAULT_CONCURRENCY = 3  # concurrent inferences per model (vLLM batches them)
# conservative install default; deployments tune it per model so a full batch of 1080p
# stills fills the context (a 1080p still costs ~2k tokens on 32 px cell models, ~2.7k on
# 28 px cells, ~3.3k on the tiled Nemotron; keep a ~4k reserve for prompt + answer -
# e.g. 60 for the Qwen3/Cosmos family at 128k, 36 for Nemotron,
# 28 for a 64k context). Targeting 4K stills instead roughly quarters the Qwen-family
# numbers; models whose pixel ceiling is below 4K (Nemotron tiles) gain no detail from
# 4K inputs.
DEFAULT_MAX_IMAGES = 16
PROMPT_RESERVE_TOKENS = 8192  # question + answer/thinking headroom inside the context


def max_images_for(cat: dict, max_model_len: int) -> int:
    """Per-request image budget: as many 1080p stills as fill the context.

    Uses the entry's measured tokens_per_image (1080p cost on its preprocessor);
    entries without one keep their explicit max_images / the conservative default.
    Capped at 120 (the deep-retrieval envelope validated on real tasks was ~80)."""
    tpi = cat.get("tokens_per_image")
    if not tpi:
        return cat.get("max_images", DEFAULT_MAX_IMAGES)
    return max(4, min(120, (max_model_len - PROMPT_RESERVE_TOKENS) // tpi))
# 96 frames (measured 2026-08-14, frames-study): num_frames is a CAP, not a quota -
# the engine samples min(cap, frames in the clip). On the Qwen3-VL/Cosmos family all
# frames of one video share a 24 Mpx budget (~12k tokens), so cost stays flat while
# per-frame detail falls as frames rise: a 1080p frame keeps ~1344x768 at 24 frames,
# ~672x384 at 96. High caps cost latency on long clips (2-3x look time on GB10).
# Temporal recall on a 2 s event stream reached 1.0 only at 96 on that family; the
# study's small-text detail test stayed 1.0 from 24 up on every model.
DEFAULT_VIDEO_FRAMES = 96
# vLLM >= 0.24 hands checkpoints whose video processor is Qwen3VLVideoProcessor to its own
# "qwen3_vl" loader, which ignores num_frames and samples a fixed 2 fps (a 6 s clip gets
# 12 frames instead of 96, a 75 s one 142). Entries on such checkpoints pin the uniform
# "opencv" loader the frames study above was measured with; dockerctl passes it as the
# media-io-kwargs video_backend (older vLLM pops that key and defaults to opencv anyway).
# A pinned loader's frames are final: that processor also resamples to 2 fps any clip the
# loader passes whole (<= cap frames: short clips, every watch chunk - watch --fps 8 got
# 2 fps, a 2 s native clip 4 of its 60 frames, on every vLLM so far), so video requests
# to these entries carry do_sample_frames=false (tasks.video_mm_kwargs; a server-wide
# --mm-processor-kwargs loses to the loader's per-clip flag).
# A model TOML may set its own video_loader ("" keeps vLLM's choice).
UNIFORM_VIDEO_LOADER = "opencv"


# request fields AISee sets per call - a sampling table must not override them
_RESERVED_SAMPLING = {"model", "messages", "max_tokens", "max_completion_tokens", "stream",
                      "stream_options", "n"}


def thinking_sampling(entry: dict) -> dict:
    """Sampling overrides for a thinking-toggle entry's thinking-on requests (merged over
    the default temperature 0.6; always-thinking reasoning entries keep their own): the
    TOML's thinking_sampling table when set, else the catalog's - decided at use, not
    frozen at install. Reserved request fields are dropped; a non-table value raises."""
    if "thinking_sampling" in entry:
        v, where = entry["thinking_sampling"], f"{entry.get('slug', '?')}.toml"
    else:
        v, where = known(entry.get("slug", "")).get("thinking_sampling"), "catalog"
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise TypeError(f"thinking_sampling in {where} must be a table, e.g. "
                         f"thinking_sampling = {{ temperature = 1.0 }} - got {v!r}")
    if not isinstance(v.get("chat_template_kwargs", {}), dict):
        raise TypeError(f"thinking_sampling.chat_template_kwargs in {where} must be a table")
    return {k: x for k, x in v.items() if k not in _RESERVED_SAMPLING}


def known(slug: str) -> dict:
    """Catalog data for an installed slug, retired entries included ({} off-catalog)."""
    return CATALOG.get(slug) or RETIRED.get(slug) or {}


def video_loader(entry: dict) -> str:
    """The vLLM video loader an installed entry pins ("" = vLLM's choice): the TOML's own
    video_loader when set, else the catalog's - decided at use, not frozen at install."""
    if "video_loader" in entry:
        return entry["video_loader"] or ""
    return known(entry.get("slug", "")).get("video_loader") or ""
DEFAULT_MAX_MODEL_LEN = 131072            # upper cap for the auto-sizing
CONTEXT_CANDIDATES = (262144, 131072, 65536, 32768, 16384, 8192)
# per-model checkpoint ceiling (max_position_embeddings); auto-sizing never exceeds it.
# Entries without ctx_native are capped at DEFAULT_MAX_MODEL_LEN.
ACTIVATION_HEADROOM_GIB = 4               # runtime overhead on top of weights + KV
# unified memory (GB10): the GPU pool IS system RAM, so a model's slice must leave room
# for the OS, AISee itself, and the small audio models. Sizing normally comes from
# mem_gib; this fraction is only the fallback for off-catalog models without one.
GPU_FRAC_UNIFIED = 0.75
UNIFIED_CAPACITY_BUDGET = 0.92  # resident gpu_frac sum cap on unified hosts (OS reserve)
GPU_FRAC_DISCRETE = 0.97  # dedicated VRAM: literal 1.0 fails vLLM's free-memory check
                          #   (driver/ECC overhead holds a few hundred MiB at startup)

# Absolute memory requirement per model (GiB), portable across GPUs: weights +
# ~4 GiB runtime + a KV pool of KV_TARGET_X full contexts (never below one full
# context). The serving fraction is derived from this at container start
# (mem_gib / detected GPU memory, clamped to the host cap), and the capacity
# check compares GiB against what is actually free. Validated on real tasks
# 2026-08-10 (see res/report-gpu-memory-budgeting.md in the project).
KV_TARGET_X = 2.5
ACT_GIB = 4  # runtime overhead next to the weights


def mem_requirement_gib(cat: dict) -> float | None:
    """weights + overhead + KV pool sized for KV_TARGET_X full contexts (floor 1.0)."""
    w, kv = cat.get("weights_gib"), cat.get("kv_gib_128k")
    if not w or not kv:
        return None
    return round(w + ACT_GIB + max(kv * KV_TARGET_X, float(kv)), 1)

CATALOG: dict[str, dict] = {
    # ---- Qwen3.5 family: natively multimodal hybrids. 3 of 4 layers are Gated DeltaNet
    # linear attention with a fixed-size state, so only every 4th layer keeps a KV cache
    # (~4-5x cheaper per token than Qwen3-VL). One checkpoint thinks or not per call via
    # the chat template's enable_thinking (thinking_toggle). Same Qwen3-VL processors
    # (32 px cells, one pixel budget per video), hence the same loader pin.
    # --reasoning-parser qwen3 is required: a thinking answer carries its chain of thought
    # before </think>; with thinking off the template closes the think block in the prompt.
    "qwen3-6-35b-a3b": {
        "hf_id": "Qwen/Qwen3.6-35B-A3B",
        "tokens_per_image": 2200,
        "ctx_native": 262144,
        "image": DEFAULT_IMAGE,
        # 10 of 40 layers hold KV (2 KV heads x 256): 20 KiB/token BF16
        "weights_gib": 67, "kv_gib_128k": 2.5,
        "mem_gib": 78,
        "extra_args": ["--reasoning-parser", "qwen3", "--kv-cache-dtype", "fp8"],
        "supports_native_video": True,
        "video_loader": UNIFORM_VIDEO_LOADER,
        "reasoning": False,
        "thinking_toggle": True,
        "load_timeout": 3600,
        "license": "Apache-2.0",
        "strengths": "Successor to both Qwen3-VL-30B-A3B checkpoints in one (MoE, ~3B active "
                     "params; thinking switches per call). Thinking off it matched or beat "
                     "Qwen3-VL-30B-A3B-Instruct on every benchmark suite (+5 points on AISee's "
                     "own items, fewer false passes) at the same per-token speed; thinking on it "
                     "beat Qwen3-VL-30B-A3B-Thinking. Only 10 of 40 layers keep a KV cache, so "
                     "256k context costs little memory.",
        "weaknesses": "Thinking (opt-in per call) is long: ~1,100 tokens per answer (1.9-3.6x "
                      "Qwen3-VL-30B-A3B-Thinking's) and ~5% of answers hit the 8192-token budget - "
                      "keep it for hard questions. Asking again about the same media re-pays the "
                      "full prefill (vLLM keeps prefix caching off for hybrid models).",
        "pitfalls": "Keep --reasoning-parser qwen3 in the serve args. First install "
                    "downloads ~72 GB.",
    },
    "qwen3-8-27b": {
        "hf_id": "Qwen/Qwen3.8-27B",
        "tokens_per_image": 2200,
        "ctx_native": 262144,
        "image": DEFAULT_IMAGE,
        # 16 of 64 layers hold KV (4 KV heads x 256): 64 KiB/token BF16
        "weights_gib": 52, "kv_gib_128k": 8,
        "mem_gib": 76,
        "extra_args": ["--reasoning-parser", "qwen3", "--kv-cache-dtype", "fp8"],
        "supports_native_video": True,
        "video_loader": UNIFORM_VIDEO_LOADER,
        "reasoning": False,
        "thinking_toggle": True,
        "load_timeout": 3600,
        "license": "Apache-2.0",
        "strengths": "Successor to Qwen3-VL-32B (dense 27B, per-call thinking toggle): same "
                     "accuracy on AISee's items, top DocVQA / ScreenSpot scores in our benchmark. "
                     "Only 16 of 64 layers do full attention, so video prefills ~3.7x faster "
                     "than the 32B's on a GB10 and 256k context fits in 76 GiB.",
        "weaknesses": "Dense: every token reads all ~52 GiB of weights - ~4.5 tok/s on a GB10 "
                      "(~25 s per still assert); thinking multiplies that. Prefer the MoE "
                      "default unless depth matters.",
        "pitfalls": "Keep --reasoning-parser qwen3 in the serve args. First install "
                    "downloads ~56 GB.",
    },
    "qwen3-5-9b": {
        "hf_id": "Qwen/Qwen3.5-9B",
        "tokens_per_image": 2200,
        "ctx_native": 262144,
        "image": DEFAULT_IMAGE,
        # 8 of 32 layers hold KV (4 KV heads x 256): 32 KiB/token BF16, plus the MTP layer's 4
        "weights_gib": 18, "kv_gib_128k": 4.5,
        "mem_gib": 32,
        # MTP: the checkpoint's own draft head proposes 2 tokens per step. Same answers and
        # verdicts on AISee's benchmark items, calls 0.78x (GB10) / 0.80x (RTX), for ~13% of
        # the KV pool and ~1 min more load. Not on the 3.6 / 27B: under the watch repetition
        # penalty vLLM 0.27.1's spec decoding drifts from plain greedy (2026-09 benchmark)
        "extra_args": ["--reasoning-parser", "qwen3", "--kv-cache-dtype", "fp8",
                       "--speculative-config", '{"method": "mtp", "num_speculative_tokens": 2}'],
        "supports_native_video": True,
        "video_loader": UNIFORM_VIDEO_LOADER,
        "reasoning": False,
        "thinking_toggle": True,
        "load_timeout": 3600,
        "license": "Apache-2.0",
        "strengths": "Small dense 9B with a per-call thinking toggle: beat Nemotron-Nano-12B-VL "
                     "by 6-12 points on every benchmark set (UI click points 0.89 vs 0.29, "
                     "75-image looks) in an 18 GiB BF16 checkpoint. Served with MTP speculative "
                     "decoding: ~21 tok/s on a GB10, ~110 on an RTX PRO 6000.",
        "weaknesses": "BF16 dense: decodes slower than a 4-bit model of its size, made up by short "
                      "answers. Thinking (opt-in per call, ~1,100 tokens per answer) costs ~1 min "
                      "per call on a GB10.",
        "pitfalls": "Keep --reasoning-parser qwen3 in the serve args. MTP costs ~13% of the KV "
                    "cache and ~1 min of load time; remove the --speculative-config pair from "
                    "extra_args (then model stop) to serve it plain. First install downloads "
                    "~19 GB.",
    },
    "cosmos-reason2-8b": {
        "hf_id": "nvidia/Cosmos-Reason2-8B",
        "tokens_per_image": 2200,
        "ctx_native": 262144,
        "image": DEFAULT_IMAGE,
        "weights_gib": 17, "kv_gib_128k": 17.5,
        "mem_gib": 66,
        "extra_args": ["--reasoning-parser", "qwen3", "--kv-cache-dtype", "fp8"],
        "supports_native_video": True,
        "video_loader": UNIFORM_VIDEO_LOADER,
        "reasoning": True,
        "load_timeout": 7200,
        "license": "NVIDIA Open Model",
        "strengths": "Purpose-built temporal / physical video reasoning; fast (~5 s asserts); "
                     "handles native video well.",
        "weaknesses": "Not a UI specialist; weaker on dense-text stills than the Qwen family.",
        "pitfalls": "Reasoning model: answers can arrive in reasoning_content with content null "
                    "(AISee falls back automatically); give it headroom in max_tokens.",
    },
    "cosmos3-nano": {
        "hf_id": "nvidia/Cosmos3-Nano",
        "tokens_per_image": 2200,
        "ctx_native": 262144,
        # multi-arch manifest (arm64 + amd64); the -aarch64 tag broke x86 hosts with
        # "exec format error"
        "image": "vllm/vllm-omni:cosmos3",

        "weights_gib": 32, "kv_gib_128k": 14.5,
        "mem_gib": 72,
        "extra_args": ["--hf-overrides", '{"architectures": ["Cosmos3ForConditionalGeneration"]}',
                       "--trust-remote-code", "--kv-cache-dtype", "fp8"],
        "supports_native_video": True,
        # a Qwen3-VL subclass with the same video processor on vLLM 0.25 (validated
        # 2026-09-26: 2 fps and no frame cap without the pin)
        "video_loader": UNIFORM_VIDEO_LOADER,
        "reasoning": True,
        "load_timeout": 5400,
        "license": "NVIDIA Open Model",
        "strengths": "Strong temporal/physical video reasoning; correct OCR; handles native video.",
        "weaknesses": "Slow to come up. The first request after its first load on a new "
                      "serving image compiles kernels (~24 s on a GB10, ~10 s on an RTX PRO "
                      "6000); they are kept, so later loads add a second or two.",
        "pitfalls": "Serves only on the vllm-omni image (multi-arch) with architecture override "
                    "Cosmos3ForConditionalGeneration; ~9-minute quiet init before weight shards "
                    "load - it is not hung.",
    },
    "cosmos3-super": {
        "hf_id": "nvidia/Cosmos3-Super",
        "tokens_per_image": 2200,
        "ctx_native": 262144,
        # reasoner-only serving: Cosmos3 is a two-tower MoT (32B AR reasoner + 32B
        # diffusion generator). Upstream vLLM's Cosmos3ForConditionalGeneration IS the
        # reasoner-only path ("the Reasoner-only part" per its source), so the 64B
        # omnimodel's understanding side fits a single 96 GB card (64k) or a GB10 (128k).
        # Needs vLLM >= 0.24 (the older cosmos3 image tag also works for nano but its
        # vLLM predates this model's config); multi-arch image (amd64 + arm64).
        "image": "vllm/vllm-omni:v0.24.0",
        "weights_gib": 64, "kv_gib_128k": 30.5,
        # 102 not 108: an idle GB10 has ~116 GiB free and the admission gate keeps a
        # 10 GiB always-free margin, so anything above ~106 could never start there.
        # 256k fp8 KV still fits: pool = 102 - weights - 4 runtime >= the 256k KV cost.
        "mem_gib": 102,
        "extra_args": ["--hf-overrides",
                       '{"architectures": ["Cosmos3ForConditionalGeneration"]}',
                       "--trust-remote-code", "--kv-cache-dtype", "fp8"],
        "supports_native_video": True,
        "video_loader": UNIFORM_VIDEO_LOADER,  # as cosmos3-nano (validated on v0.24.0)
        "reasoning": True,
        "load_timeout": 10800,
        "license": "NVIDIA Open Model",
        "strengths": "The 64B omnimodel's Reasoner tower (32B): deepest physical/temporal "
                     "reasoning in the catalog; correct dense OCR; handles native video. "
                     "Fast for its size (~3-7 s stills / asserts measured on RTX PRO 6000).",
        "weaknesses": "128k context on 96 GB cards (256k needs a GB10-class pool). "
                      "Borderline UI-state asserts can flip between runs - phrase "
                      "expectations concretely.",
        "pitfalls": "Understanding only - the generator tower is not loaded, so no "
                    "image/video generation. First install downloads the full ~130 GB "
                    "checkpoint although only the reasoner half loads. Requires a "
                    "vLLM >= 0.24 serving image.",
    },
    # ---- audio models (engine != vllm; modality "audio") ----
    # Serving images are built locally on the host from res/serving/<engine dir>/
    # the first time the model starts. gpu_frac values are honest small fractions so
    # the capacity check allows co-residency next to a resident VLM.
    "parakeet-tdt-0-6b-v3": {
        "hf_id": "nvidia/parakeet-tdt-0.6b-v3",
        "image": "aisee/audio-nemo:1",
        "engine": "nemo-asr",
        "modality": "audio",
        "capabilities": ["transcribe"],
        "gpu_frac": 0.06,   # ~5 GB resident (transducer: no KV cache)
        "mem_gib": 7,       # steady-state; long-form transients bounded by mem_limit
        # long-form ASR peaks high in HOST RAM: timestamp alignment on a 79-min file
        # hit ~16 GB anon rss (measured); the cap is the contained-failure bound
        "mem_limit": "40g",
        "concurrency": 1,   # one GPU job at a time (unified-memory discipline)
        "load_timeout": 5400,  # first start may build the serving image on the host
        "license": "CC-BY-4.0",
        "strengths": "Recommended ASR default. 25 languages incl. Russian/Ukrainian; native "
                     "word/segment timestamps; 12-87x realtime on GB10; zero hallucination "
                     "loops on crowded/degraded meeting audio where Whisper variants looped.",
        "weaknesses": "No speaker labels by itself (pair with the diarization model); "
                      "language is auto-detected per file, not per segment.",
        "pitfalls": "Inputs longer than ~24 min automatically switch the model to local "
                    "attention (handled by the serving app). First start builds the serving "
                    "image (~10-20 min one-time).",
    },
    "pyannote-diarization-3-1": {
        "hf_id": "pyannote/speaker-diarization-3.1",
        "image": "aisee/audio-pyannote:1",
        "engine": "pyannote",
        "modality": "audio",
        "capabilities": ["diarize"],
        "gpu_frac": 0.04,   # ~3 GB resident
        "mem_gib": 4,
        "mem_limit": "16g",
        "concurrency": 1,
        "load_timeout": 5400,
        "license": "MIT (weights HF-gated by a contact form)",
        "strengths": "Recommended diarization default. Unbounded speaker count (found 9 on "
                     "an ~8-speaker meeting); ~25x realtime on GB10.",
        "weaknesses": "Over-splits long multi-party recordings (pass min/max speaker hints "
                      "when known); fails toward visible over-counting, not silent merging.",
        "pitfalls": "HF token must have accepted the licenses on THREE gated repos "
                    "(speaker-diarization-3.1, segmentation-3.0, wespeaker-voxceleb). "
                    "First start builds the serving image (~10-20 min one-time).",
    },
}

# Retired from the catalog: no longer listed or recommended, but a host that has one
# installed keeps serving it exactly as before - the at-use lookups (video loader,
# weights, describe) go through known(), and installing one by slug still works with a
# note that names its successor. 1.1.0b1: the Qwen3.5 family replaced these
# (res/report-model-refresh-qwen35.md in the project).
RETIRED: dict[str, dict] = {
    "qwen3-vl-30b-a3b-instruct": {
        "hf_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "retired_in": "1.1.0b1", "successor": "qwen3-6-35b-a3b",
        "tokens_per_image": 2200,
        "ctx_native": 262144,
        "image": DEFAULT_IMAGE,
        "weights_gib": 62, "kv_gib_128k": 10.5,
        "mem_gib": 92,
        # NOTE: Qwen3-VL has no hybrid thinking toggle - the Instruct checkpoints never
        # think (their chat template has no enable_thinking); the Thinking variants are
        # separate checkpoints. Do not add --reasoning-parser here: with a non-thinking
        # model it misroutes the whole answer into reasoning_content.
        # fp8 KV cache: halves KV cost -> 256k context in the same GiB slice; validated
        # 2026-08-10 (exact OCR, 150k-token needle retrieval 3/3, watch) on both hosts
        "extra_args": ["--kv-cache-dtype", "fp8"],
        "supports_native_video": True,
        "video_loader": UNIFORM_VIDEO_LOADER,
        "reasoning": False,
        "load_timeout": 3600,
        "license": "Apache-2.0",
        "strengths": "The default until 1.1.0b1. 32B-class quality at small-model speed (MoE, ~3B active "
                     "params): ~5-7 s stills, correct OCR on dense numbers, handles native video, "
                     "fast element grounding (~1.3 s).",
        "weaknesses": "The full ~62 GB of BF16 weights must be resident despite the speed; not a "
                      "specialist at physical/temporal reasoning.",
        "pitfalls": "Needs --enforce-eager on GB10-class hardware. First install downloads ~62 GB.",
    },
    "qwen3-vl-30b-a3b-thinking": {
        "hf_id": "Qwen/Qwen3-VL-30B-A3B-Thinking",
        "retired_in": "1.1.0b1", "successor": "qwen3-6-35b-a3b",
        "tokens_per_image": 2200,
        "ctx_native": 262144,
        "image": DEFAULT_IMAGE,
        "weights_gib": 62, "kv_gib_128k": 10.5,
        "mem_gib": 92,
        # the parser routes the always-on chain-of-thought into the reasoning field so
        # answers stay clean; without it the CoT would pollute the answer text
        "extra_args": ["--reasoning-parser", "qwen3", "--kv-cache-dtype", "fp8"],
        "supports_native_video": True,
        "video_loader": UNIFORM_VIDEO_LOADER,
        "reasoning": True,
        "load_timeout": 7200,
        "license": "Apache-2.0",
        "strengths": "The thinking twin of the recommended default (MoE, ~3B active params): "
                     "genuine chain-of-thought with clean answers, still fast (~6 s looks, "
                     "~9 s asserts measured on GB10); exact dense OCR; handles native video; "
                     "correctly reports static clips instead of inventing motion.",
        "weaknesses": "Always thinks - cannot be disabled, so trivial checks pay the CoT tax; "
                      "thinking spends the same max_tokens budget (defaults to 8192 everywhere).",
        "pitfalls": "Keep --reasoning-parser qwen3 in the serve args or the CoT lands in the "
                    "answer text. First install downloads ~62 GB.",
    },
    "qwen3-vl-32b-instruct": {
        "hf_id": "Qwen/Qwen3-VL-32B-Instruct",
        "retired_in": "1.1.0b1", "successor": "qwen3-8-27b",
        "tokens_per_image": 2200,
        "ctx_native": 262144,
        "image": DEFAULT_IMAGE,

        "weights_gib": 63, "kv_gib_128k": 34,
        # 102 not 108: an idle GB10 has ~116 GiB free and the admission gate keeps a
        # 10 GiB always-free margin, so anything above ~106 could never start there.
        # 256k fp8 KV still fits: pool = 102 - weights - 4 runtime >= the 256k KV cost.
        "mem_gib": 102,
        "extra_args": ["--kv-cache-dtype", "fp8"],
        "supports_native_video": True,
        "video_loader": UNIFORM_VIDEO_LOADER,
        "reasoning": False,
        "load_timeout": 3600,
        "license": "Apache-2.0",
        "strengths": "Deepest synthesis / long narration; correct OCR; handles native video.",
        "weaknesses": "4-9x slower than small/MoE models on bandwidth-bound GPUs (24-45 s per still "
                      "assert). Use only when maximum reasoning depth matters.",
        "pitfalls": "gpu_frac below ~0.70 crash-loops ('No available memory for the cache blocks').",
    },
    "nvidia-nemotron-nano-12b-v2-vl-nvfp4-qad": {
        "hf_id": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD",
        "retired_in": "1.1.0b1", "successor": "qwen3-5-9b",
        "tokens_per_image": 3300,
        "ctx_native": 131072,
        # frames-study 2026-08-14: saturates at 64 (temporal 1.0, best motion count);
        # 96 adds nothing and overshoots motion counting
        "video_frames": 64,
        "image": DEFAULT_IMAGE,
        "weights_gib": 11, "kv_gib_128k": 5,
        "mem_gib": 28,
        "extra_args": ["--trust-remote-code"],
        "supports_native_video": True,
        "reasoning": False,
        "load_timeout": 7200,
        "license": "NVIDIA Open Model (commercial use permitted)",
        "strengths": "Fastest overall (NVFP4, ~11 GB resident): ~4-7 s stills, ~1 s OCR/grounding; "
                     "handles native video; smallest GPU footprint.",
        "weaknesses": "Fumbled a dense number in testing (OCR digit slip) - do not trust it for "
                      "exact figures.",
        "pitfalls": "Needs --trust-remote-code and --enforce-eager. NVFP4 quantization is "
                    "auto-detected - do NOT pass --quantization. On a GB10 the first request "
                    "after its first load on a new serving image takes up to ~2.5 min "
                    "(one-time kernel JIT, cached for later loads); later calls take seconds.",
    },
}

RECOMMENDED_DEFAULT = "qwen3-6-35b-a3b"

# capability -> task kinds it serves (vision models carry no capabilities field and
# implicitly serve look/assert/watch)
VISION_KINDS = ("look", "assert", "watch")
AUDIO_KINDS = ("transcribe", "diarize")


def slugify(model_name: str) -> str:
    """Slug of the model name with the org prefix dropped: Qwen/Qwen3-VL-32B-Instruct -> qwen3-vl-32b-instruct."""
    import re
    name = model_name.split("/")[-1].lower()
    return re.sub(r"-+$", "", re.sub(r"^-+", "", re.sub(r"[^a-z0-9]+", "-", name)))


def lookup(name: str) -> tuple[str, dict | None]:
    """Resolve a catalog slug or HF id to (slug, catalog entry or None); retired entries
    resolve too, so reinstalling one keeps its serving flags (see RETIRED)."""
    both = {**RETIRED, **CATALOG}
    if name in both:
        return name, both[name]
    slug = slugify(name)
    if slug in both:
        return slug, both[slug]
    for s, e in both.items():
        if e["hf_id"].lower() == name.lower():
            return s, e
    return slug, None
