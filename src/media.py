# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Media pipeline: ffmpeg/ffprobe frame sampling, segment re-encode, base64 packaging.

Known pitfalls honored here:
- Even sampling must use fps=N/duration (ffprobe), not fps=1 + -frames:v N.
- Native video is re-encoded to MJPEG-in-AVI: serving containers commonly ship an OpenCV/ffmpeg
  without an H.264 decoder, while intra-frame MJPEG decodes everywhere.
"""

import base64
import mimetypes
import os
import shutil
import subprocess
from pathlib import Path

VIDEO_EXT = {".mov", ".mp4", ".webm", ".mkv", ".avi", ".gif"}


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXT


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found on the host; run `aisee install`")


def video_duration(path: str | Path) -> float | None:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            text=True, stderr=subprocess.DEVNULL).strip()
        dur = float(out)
        return dur if dur > 0 else None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def sample_even(path: str | Path, frames: int, out_dir: Path) -> list[Path]:
    """`frames` frames spread evenly across the WHOLE clip."""
    require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "f_%03d.png")
    dur = video_duration(path)
    vf = f"fps={max(frames, 1)}/{dur:.3f}" if dur else "fps=1"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path),
                    "-vf", vf, "-frames:v", str(frames), "-fps_mode", "vfr", pattern], check=True)
    out = sorted(out_dir.glob("f_*.png"))
    if not out:
        raise RuntimeError(f"ffmpeg produced no frames from {path}")
    return out


def sample_fps(path: str | Path, fps: float, out_dir: Path,
               max_frames: int | None = None) -> list[Path]:
    """Frames at a fixed rate (frames per second of source time)."""
    require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "f_%04d.png")
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-vf", f"fps={fps}"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-fps_mode", "vfr", pattern]
    subprocess.run(cmd, check=True)
    out = sorted(out_dir.glob("f_*.png"))
    if not out:
        raise RuntimeError(f"ffmpeg produced no frames from {path}")
    return out


def reencode_segment(path: str | Path, start: float, dur: float, fps: float | None,
                     scale: int | None, out_dir: Path, tag: str = "seg") -> Path:
    """Cut [start, start+dur) to an MJPEG-AVI clip, optionally resampled/downscaled."""
    require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}_{start:.1f}.avi"
    vf = []
    if fps:
        vf.append(f"fps={fps}")
    if scale:
        vf.append(f"scale=-2:{scale}")
    cmd = ["ffmpeg", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(path)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-an", "-c:v", "mjpeg", "-q:v", "6", str(out)]
    subprocess.run(cmd, check=True)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no output for segment {start:.1f}s+{dur:.1f}s "
                           f"of {path} - the window may contain no frames")
    return out


def img_data_url(path: str | Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def video_data_url(path: str | Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("video/"):
        mime = "video/mp4"
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_content(media: list[str], text: str, *, frames: int, fps: float | None,
                  native: bool, max_images: int, work_dir: Path) -> list[dict]:
    """OpenAI-style content parts for one user message: media (images / sampled or native video) + text."""
    parts: list[dict] = []
    n_images = 0
    for m in media:
        if is_video(m) and native:
            # send the video itself; always re-encode so the serving container can decode it
            src = reencode_segment(m, 0.0, video_duration(m) or 3600.0, fps, None,
                                   work_dir, tag=Path(m).stem)
            parts.append({"type": "video_url", "video_url": {"url": video_data_url(src)}})
        elif is_video(m):
            imgs = (sample_fps(m, fps, work_dir / f"{Path(m).stem}-frames")
                    if fps else sample_even(m, frames, work_dir / f"{Path(m).stem}-frames"))
            for img in imgs:
                if n_images >= max_images:
                    break
                parts.append({"type": "image_url", "image_url": {"url": img_data_url(img)}})
                n_images += 1
        else:
            if n_images >= max_images:
                raise ValueError(f"more than {max_images} images for this model (server cap)")
            parts.append({"type": "image_url", "image_url": {"url": img_data_url(m)}})
            n_images += 1
    parts.append({"type": "text", "text": text})
    return parts


def thumbnail(src: str | Path, dest: Path, width: int = 240) -> Path:
    """First-frame JPEG thumbnail for an image or video (cached by the caller)."""
    require_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
                    "-vf", f"scale={width}:-2", "-frames:v", "1", str(dest)], check=True)
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"could not thumbnail {src}")
    return dest


def stage_bytes(data: bytes, filename: str, dest_dir: Path) -> Path:
    """Save an uploaded file into the task's staging dir with a sanitized name."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = os.path.basename(filename) or "upload.bin"
    p = dest_dir / safe
    p.write_bytes(data)
    return p
