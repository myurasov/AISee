# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Per-model input resolution facts for /v1/describe.

AISee never downscales by default (look extracts native-resolution frames; only watch's
optional scale param resizes client-side), so the model's own preprocessor is the only
implicit resizer. This module derives the exact still/video pixel budgets from the model's
HF snapshot on disk (preprocessor_config.json / video_preprocessor_config.json) plus the
serving config in effect, so `describe` can tell a consumer whether e.g. a 1080p screen
recording is read at native resolution or silently downscaled.

Known preprocessor schemes:
- Qwen3-VL style (Qwen3VLProcessor: Qwen3-VL, Cosmos3, Cosmos-Reason2): pixel budgets in
  size.shortest_edge/longest_edge; the VIDEO budget is TOTAL across all sampled frames.
- Qwen2-VL style (Qwen2_5_VLProcessor: Holo1.5, UI-TARS): min_pixels/max_pixels (or the
  same size keys); the video budget applies PER FRAME.
- Nemotron tile style (NemotronNanoVLV2*): fixed image_size tiles, up to max_num_tiles
  (+ optional thumbnail tile).
Anything else is reported as unknown rather than guessed. When the snapshot is not yet
downloaded, catalog fallbacks (same numbers, hand-copied) are used and marked estimated.
"""

import json
from pathlib import Path

from . import catalog, paths

# standard resolutions used for the worked examples in the guide (largest first)
_STANDARD_RES = [("8K", 7680, 4320), ("4K", 3840, 2160), ("1440p", 2560, 1440),
                 ("1080p", 1920, 1080), ("720p", 1280, 720), ("480p", 854, 480),
                 ("360p", 640, 360), ("240p", 426, 240)]

# answer + prompt headroom subtracted from the context before dividing it among frames
_CTX_OVERHEAD_TOKENS = 2048

# pre-download fallbacks (values copied from the published preprocessor configs);
# used only when the local snapshot is missing, and always marked estimated
_CATALOG_FALLBACKS: dict[str, dict] = {
    "qwen3-vl-30b-a3b-instruct": {"scheme": "qwen3", "patch": 16, "merge": 2, "tps": 2,
                                  "still_min": 65536, "still_max": 16777216,
                                  "video_total": 25165824},
    "qwen3-vl-32b-instruct": {"scheme": "qwen3", "patch": 16, "merge": 2, "tps": 2,
                              "still_min": 65536, "still_max": 16777216,
                              "video_total": 25165824},
    "cosmos-reason2-8b": {"scheme": "qwen3", "patch": 16, "merge": 2, "tps": 2,
                          "still_min": 65536, "still_max": 16777216,
                          "video_total": 25165824},
    "cosmos3-nano": {"scheme": "qwen3", "patch": 16, "merge": 2, "tps": 2,
                     "still_min": 65536, "still_max": 16777216, "video_total": 25165824},
    "cosmos3-super": {"scheme": "qwen3", "patch": 16, "merge": 2, "tps": 2,
                      "still_min": 65536, "still_max": 16777216, "video_total": 25165824},
    "holo1-5-7b": {"scheme": "qwen2", "patch": 14, "merge": 2, "tps": 2,
                   "still_min": 3136, "still_max": 3686400, "video_frame": 3686400},
    "ui-tars-1-5-7b": {"scheme": "qwen2", "patch": 14, "merge": 2, "tps": 2,
                       "still_min": 3136, "still_max": 12845056, "video_frame": 12845056},
    "nvidia-nemotron-nano-12b-v2-vl-nvfp4-qad": {"scheme": "tiles", "tile_px": 512,
                                                 "patch": 16, "downsample": 0.5,
                                                 "max_tiles": 12, "thumbnail": True},
}


def _snapshot_dir(hf_id: str) -> Path | None:
    base = paths.hf_cache() / "hub" / ("models--" + hf_id.replace("/", "--")) / "snapshots"
    if not base.is_dir():
        return None
    snaps = [d for d in base.iterdir() if (d / "preprocessor_config.json").exists()]
    return max(snaps, key=lambda d: d.stat().st_mtime) if snaps else None


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _pixel_budget(cfg: dict) -> tuple[int | None, int | None]:
    """(min_pixels, max_pixels) from either qwen key style."""
    size = cfg.get("size") or {}
    lo = cfg.get("min_pixels", size.get("shortest_edge"))
    hi = cfg.get("max_pixels", size.get("longest_edge"))
    return lo, hi


def _parse_snapshot(snap: Path) -> dict | None:
    """Normalize a snapshot's preprocessor configs to the internal scheme dict."""
    cfg = _read_json(snap / "preprocessor_config.json")
    if not cfg:
        return None
    proc = str(cfg.get("image_processor_type") or "")
    if "Nemotron" in proc:
        if not cfg.get("image_size") or not cfg.get("max_num_tiles"):
            return None
        return {"scheme": "tiles", "tile_px": int(cfg["image_size"]),
                "patch": int(cfg.get("patch_size") or 16),
                "downsample": float(cfg.get("downsample_ratio") or 1.0),
                "max_tiles": int(cfg["max_num_tiles"]),
                "thumbnail": bool(cfg.get("use_thumbnail"))}
    if "Qwen2VLImageProcessor" in proc:  # shared by the Qwen2- and Qwen3-VL families
        lo, hi = _pixel_budget(cfg)
        if not cfg.get("patch_size") or not hi:
            return None
        out = {"patch": int(cfg["patch_size"]), "merge": int(cfg.get("merge_size") or 2),
               "tps": int(cfg.get("temporal_patch_size") or 2),
               "still_min": lo, "still_max": hi}
        vcfg = _read_json(snap / "video_preprocessor_config.json")
        vbudget = _pixel_budget(vcfg)[1] if vcfg else None
        # Qwen3-family video processors budget TOTAL pixels across all sampled frames;
        # Qwen2-family ones budget each frame independently
        if "Qwen3VL" in str((vcfg or cfg).get("processor_class") or "") or \
           "Qwen3VL" in str((vcfg or {}).get("video_processor_type") or ""):
            out.update({"scheme": "qwen3", "video_total": vbudget})
        else:
            out.update({"scheme": "qwen2", "video_frame": vbudget or hi})
        return out
    return None


def _mp(px: int | None) -> str:
    return f"{px / 1e6:.2f} MP" if px else "unknown"


def _std_example(max_px: int | None, unit: str) -> str:
    """Anchor a pixel budget to the standard-resolution ladder (8K/4K/1080p/...)."""
    if not max_px:
        return "unknown"
    fits = next(((n, w, h) for n, w, h in _STANDARD_RES if w * h <= max_px), None)
    if fits is None:
        return f"even a 240p {unit} is downscaled"
    name, w, h = fits
    if fits == _STANDARD_RES[0]:
        return f"an 8K {unit} ({w}x{h}) passes untouched"
    # the smallest standard resolution ABOVE the budget, with its downscaled size
    on, ow, oh = _STANDARD_RES[_STANDARD_RES.index(fits) - 1]
    scale = (max_px / (ow * oh)) ** 0.5
    return (f"{name} ({w}x{h}) and below pass untouched; {on} is downscaled to "
            f"~{round(ow * scale)}x{round(oh * scale)}")


def _example_still(max_px: int | None) -> str:
    return _std_example(max_px, "still")


def _example_frame(per_frame_px: int | None) -> str:
    return _std_example(per_frame_px, "frame")


def input_resolution(entry: dict) -> dict:
    """Still/video input-resolution facts for one installed-model registry entry."""
    snap = _snapshot_dir(entry["hf_id"])
    scheme = _parse_snapshot(snap) if snap else None
    source = "preprocessor_config"
    if scheme is None:
        scheme = _CATALOG_FALLBACKS.get(entry["slug"])
        source = "catalog-fallback" if scheme else "unknown"
        if scheme and snap:
            # snapshot exists but the preprocessor is exotic: honesty over guessing
            scheme, source = None, "unknown"
    estimated = source != "preprocessor_config"
    if scheme is None:
        return {"still": "unknown", "video": "unknown",
                "estimated": True, "source": source}

    frames = int(entry.get("video_frames") or catalog.DEFAULT_VIDEO_FRAMES)
    ctx = int(entry.get("max_model_len") or 0)
    stills_only = not entry.get("supports_native_video", True)

    if scheme["scheme"] == "tiles":
        tiles = scheme["max_tiles"] + (1 if scheme.get("thumbnail") else 0)
        max_px = scheme["tile_px"] ** 2 * tiles
        tokens_per_tile = int((scheme["tile_px"] / scheme["patch"]) ** 2
                              * scheme["downsample"] ** 2)
        still = {"scheme": "tiles", "tile_px": scheme["tile_px"], "max_tiles": tiles,
                 "max_pixels": max_px, "max_megapixels": round(max_px / 1e6, 2),
                 "min_pixels": None, "patch_px": scheme["patch"],
                 "aspect_preserved": True,  # tiling pads/arranges, gross aspect kept
                 "tokens_per_tile": tokens_per_tile, "example": _example_still(max_px)}
        # per-frame tiling for video is processor-internal; do not guess
        video = ({"stills_only": True,
                  "note": "video is read as a single frame at the still budget"}
                 if stills_only else
                 {"per_frame_max_pixels": None, "frame_budget": frames,
                  "stills_only": False, "example": "unknown"})
        return {"still": still, "video": video, "estimated": estimated, "source": source}

    eff = scheme["patch"] * scheme["merge"]          # px per token cell edge
    px_per_token_still = eff * eff
    still_max = scheme.get("still_max")
    still = {"scheme": scheme["scheme"], "patch_px": scheme["patch"],
             "token_cell_px": eff, "min_pixels": scheme.get("still_min"),
             "max_pixels": still_max,
             "max_megapixels": round(still_max / 1e6, 2) if still_max else None,
             "aspect_preserved": True,
             "example": _example_still(still_max)}

    if stills_only:
        video = {"stills_only": True,
                 "note": "video is read as a single frame at the still budget"}
    else:
        # a video token covers one token cell across temporal_patch_size frames
        px_per_token_video = px_per_token_still * scheme.get("tps", 2)
        if scheme["scheme"] == "qwen3":
            total = scheme.get("video_total")
            per_frame_cfg = int(total / frames) if total else None
        else:
            total = None
            per_frame_cfg = scheme.get("video_frame")
        per_frame_ctx = (int((ctx - _CTX_OVERHEAD_TOKENS) / frames * px_per_token_video)
                         if ctx > _CTX_OVERHEAD_TOKENS else None)
        candidates = [p for p in (per_frame_cfg, per_frame_ctx) if p]
        per_frame = min(candidates) if candidates else None
        video = {"per_frame_max_pixels": per_frame, "frame_budget": frames,
                 "total_pixel_budget": total,
                 "context_bound": bool(per_frame_ctx and per_frame == per_frame_ctx
                                       and per_frame != per_frame_cfg),
                 "stills_only": False, "example": _example_frame(per_frame)}

    return {"still": still, "video": video, "estimated": estimated, "source": source}


def markdown_line(ir: dict) -> str:
    """One `Input resolution:` guide line from an input_resolution() dict."""
    tag = (" (estimated: weights not downloaded yet)"
           if ir.get("source") == "catalog-fallback" else "")
    still, video = ir.get("still"), ir.get("video")
    if still == "unknown" or not isinstance(still, dict):
        return "- Input resolution: unknown (unrecognized preprocessor)"
    if still.get("scheme") == "tiles":
        s = (f"stills tiled into up to {still['max_tiles']} tiles of "
             f"{still['tile_px']}x{still['tile_px']} px (~{still['max_megapixels']} MP; "
             f"{still['example']})")
    else:
        s = (f"stills up to {_mp(still['max_pixels'])} at native resolution, "
             f"{still['token_cell_px']} px token cells, aspect preserved "
             f"({still['example']})")
    if isinstance(video, dict) and video.get("stills_only"):
        v = "video is read as a SINGLE frame at the still budget"
    elif isinstance(video, dict) and video.get("per_frame_max_pixels"):
        shared = (video.get("total_pixel_budget")
                  and "; the budget is shared, so fewer sampled frames get "
                      "proportionally more pixels each" or "")
        v = (f"video ~{_mp(video['per_frame_max_pixels'])} per frame at "
             f"{video['frame_budget']} frames"
             + (" (context-bound)" if video.get("context_bound") else "")
             + f" ({video['example']}{shared})")
    else:
        v = "video per-frame budget unknown"
    return f"- Input resolution: {s}; {v}.{tag}"
