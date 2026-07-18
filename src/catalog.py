# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Seed model catalog: install by slug without hand-writing serving flags.

Entries carry serving requirements plus agent-facing strengths / weaknesses / pitfalls
(measured on a DGX Spark GB10, 2026-07) that feed the /v1/describe model guide.
"""

DEFAULT_IMAGE = "nvcr.io/nvidia/vllm:26.06-py3"

# Serving defaults assume the main mode of operation: ONE model resident per GPU, and are
# computed from the detected GPU at install time (registry.gpu_profile / fit_max_model_len):
# gpu_frac is ~1.0 on discrete GPUs and 0.90 on unified-memory systems (GB10 class, where
# the GPU pool is also system RAM); max_model_len is the largest standard context whose
# KV cache fits next to the weights (catalog entries carry weights_gib / kv_gib_128k
# estimates). Known tiers: GB10 (~120 GiB unified), 96 GB and 48 GB discrete.
DEFAULT_CONCURRENCY = 3  # concurrent inferences per model (vLLM batches them)
# conservative install default; deployments tune it per model so a full batch of 1080p
# stills fills the context (a 1080p still costs ~2k tokens on 32 px cell models, ~2.7k on
# 28 px cells, ~3.3k on the tiled Nemotron; keep a ~4k reserve for prompt + answer -
# e.g. 60 for the Qwen3/Cosmos family at 128k, 46 for Holo/UI-TARS, 36 for Nemotron,
# 28 for a 64k context). Targeting 4K stills instead roughly quarters the Qwen-family
# numbers; models whose pixel ceiling is below 4K (Holo 3.69 MP, Nemotron tiles) gain
# no detail from 4K inputs.
DEFAULT_MAX_IMAGES = 16
# 24 frames: the Qwen3-VL family budgets ~24 MP across ALL sampled frames, so 24 frames
# keep each frame at ~1.05 MP - 720p passes natively (64 frames would drop it to ~0.39 MP)
DEFAULT_VIDEO_FRAMES = 24
DEFAULT_MAX_MODEL_LEN = 131072            # upper cap for the auto-sizing
CONTEXT_CANDIDATES = (131072, 65536, 32768, 16384, 8192)
ACTIVATION_HEADROOM_GIB = 4               # runtime overhead on top of weights + KV
GPU_FRAC_UNIFIED = 0.90   # unified memory (GB10): leave headroom for system processes
GPU_FRAC_DISCRETE = 0.97  # dedicated VRAM: literal 1.0 fails vLLM's free-memory check
                          #   (driver/ECC overhead holds a few hundred MiB at startup)

CATALOG: dict[str, dict] = {
    "qwen3-vl-30b-a3b-instruct": {
        "hf_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "image": DEFAULT_IMAGE,
        "weights_gib": 62, "kv_gib_128k": 13,
        "extra_args": [],
        "supports_native_video": True,
        "reasoning": False,
        "load_timeout": 3600,
        "license": "Apache-2.0",
        "strengths": "Recommended default. 32B-class quality at small-model speed (MoE, ~3B active "
                     "params): ~5-7 s stills, correct OCR on dense numbers, handles native video, "
                     "fast element grounding (~1.3 s).",
        "weaknesses": "The full ~62 GB of BF16 weights must be resident despite the speed; not a "
                      "specialist at physical/temporal reasoning.",
        "pitfalls": "Needs --enforce-eager on GB10-class hardware. First install downloads ~62 GB.",
    },
    "qwen3-vl-32b-instruct": {
        "hf_id": "Qwen/Qwen3-VL-32B-Instruct",
        "image": DEFAULT_IMAGE,
        
        "weights_gib": 63, "kv_gib_128k": 34,
        "extra_args": [],
        "supports_native_video": True,
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
        "image": DEFAULT_IMAGE,
        "weights_gib": 11, "kv_gib_128k": 5,
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
                    "auto-detected - do NOT pass --quantization.",
    },
    "holo1-5-7b": {
        "hf_id": "Hcompany/Holo1.5-7B",
        "image": DEFAULT_IMAGE,
        "weights_gib": 16, "kv_gib_128k": 7,
        "extra_args": [],
        "supports_native_video": False,
        "reasoning": False,
        "load_timeout": 7200,
        "license": "Apache-2.0",
        "strengths": "Pixel-precise UI element grounding (computer-use lineage); very fast stills "
                     "(~1.4 s OCR, ~2.4 s grounding); low memory (~16 GB).",
        "weaknesses": "Stills-only: reads a video clip as a single frame. Terse answers.",
        "pitfalls": "Hangs during CUDA-graph capture unless served with --enforce-eager.",
    },
    "cosmos-reason2-8b": {
        "hf_id": "nvidia/Cosmos-Reason2-8B",
        "image": DEFAULT_IMAGE,
        "weights_gib": 17, "kv_gib_128k": 18,
        "extra_args": ["--reasoning-parser", "qwen3"],
        "supports_native_video": True,
        "reasoning": True,
        "load_timeout": 7200,
        "license": "NVIDIA Open Model",
        "strengths": "Purpose-built temporal / physical video reasoning; fast (~5 s asserts); "
                     "handles native video well.",
        "weaknesses": "Not a UI specialist; weaker on dense-text stills than the Qwen/Holo family.",
        "pitfalls": "Reasoning model: answers can arrive in reasoning_content with content null "
                    "(AISee falls back automatically); give it headroom in max_tokens.",
    },
    "cosmos3-nano": {
        "hf_id": "nvidia/Cosmos3-Nano",
        # multi-arch manifest (arm64 + amd64); the -aarch64 tag broke x86 hosts with
        # "exec format error"
        "image": "vllm/vllm-omni:cosmos3",

        "weights_gib": 32, "kv_gib_128k": 26,
        "extra_args": ["--hf-overrides", '{"architectures": ["Cosmos3ForConditionalGeneration"]}',
                       "--trust-remote-code"],
        "supports_native_video": True,
        "reasoning": True,
        "load_timeout": 5400,
        "license": "NVIDIA Open Model",
        "strengths": "Strong temporal/physical video reasoning; correct OCR; handles native video.",
        "weaknesses": "Slow to come up; one-time ~59 s first-call warmup after load.",
        "pitfalls": "Serves only on the vllm-omni image (multi-arch) with architecture override "
                    "Cosmos3ForConditionalGeneration; ~9-minute quiet init before weight shards "
                    "load - it is not hung.",
    },
    "cosmos3-super": {
        "hf_id": "nvidia/Cosmos3-Super",
        # reasoner-only serving: Cosmos3 is a two-tower MoT (32B AR reasoner + 32B
        # diffusion generator). Upstream vLLM's Cosmos3ForConditionalGeneration IS the
        # reasoner-only path ("the Reasoner-only part" per its source), so the 64B
        # omnimodel's understanding side fits a single 96 GB card (64k) or a GB10 (128k).
        # Needs vLLM >= 0.24 (the older cosmos3 image tag also works for nano but its
        # vLLM predates this model's config); multi-arch image (amd64 + arm64).
        "image": "vllm/vllm-omni:v0.24.0",
        "weights_gib": 64, "kv_gib_128k": 32,
        "extra_args": ["--hf-overrides",
                       '{"architectures": ["Cosmos3ForConditionalGeneration"]}',
                       "--trust-remote-code"],
        "supports_native_video": True,
        "reasoning": True,
        "load_timeout": 10800,
        "license": "NVIDIA Open Model",
        "strengths": "The 64B omnimodel's Reasoner tower (32B): deepest physical/temporal "
                     "reasoning in the catalog; correct dense OCR; handles native video. "
                     "Fast for its size (~3-7 s stills / asserts measured on RTX PRO 6000).",
        "weaknesses": "64k context on 96 GB cards (128k needs a GB10-class pool). "
                      "Borderline UI-state asserts can flip between runs - phrase "
                      "expectations concretely.",
        "pitfalls": "Understanding only - the generator tower is not loaded, so no "
                    "image/video generation. First install downloads the full ~130 GB "
                    "checkpoint although only the reasoner half loads. Requires a "
                    "vLLM >= 0.24 serving image. One-time ~35 s first-call warmup "
                    "after load.",
    },
    "ui-tars-1-5-7b": {
        "hf_id": "ByteDance-Seed/UI-TARS-1.5-7B",
        "image": DEFAULT_IMAGE,
        "weights_gib": 16, "kv_gib_128k": 7,
        "extra_args": ["--trust-remote-code"],
        "supports_native_video": False,
        "reasoning": False,
        "load_timeout": 7200,
        "license": "Apache-2.0",
        "strengths": "GUI-agent lineage: can emit click/type actions (future action generation); "
                     "correct OCR; solid still judgments.",
        "weaknesses": "Stills-only: reads a video clip as a single frame.",
        "pitfalls": "Needs --trust-remote-code and --enforce-eager.",
    },
}

RECOMMENDED_DEFAULT = "qwen3-vl-30b-a3b-instruct"


def slugify(model_name: str) -> str:
    """Slug of the model name with the org prefix dropped: Qwen/Qwen3-VL-32B-Instruct -> qwen3-vl-32b-instruct."""
    import re
    name = model_name.split("/")[-1].lower()
    return re.sub(r"-+$", "", re.sub(r"^-+", "", re.sub(r"[^a-z0-9]+", "-", name)))


def lookup(name: str) -> tuple[str, dict | None]:
    """Resolve a catalog slug or HF id to (slug, catalog entry or None)."""
    if name in CATALOG:
        return name, CATALOG[name]
    slug = slugify(name)
    if slug in CATALOG:
        return slug, CATALOG[slug]
    for s, e in CATALOG.items():
        if e["hf_id"].lower() == name.lower():
            return s, e
    return slug, None
