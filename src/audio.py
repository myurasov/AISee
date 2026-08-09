# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Audio pipeline: track probing/extraction (audio files and video containers),
multi-track handling, ASR/diarization engine clients, word->speaker merge, and
transcript artifacts (json/txt/srt/vtt).

Design notes (see res/add-audio-implementation-report.md in the project):
- Serving containers receive 16 kHz mono WAV over multipart HTTP (localhost), so
  they never need access to host paths or the task media dir.
- A multi-track container (Zoom/OBS per-participant recordings) is transcribed
  per track: each track IS one speaker, which beats any diarizer. Duplicate
  tracks (mixdown copies) are detected by waveform correlation first.
- Segment/word timestamps are absolute positions in the recording.
"""

import json
import struct
import subprocess
from pathlib import Path

import httpx


# ---------------- probing + extraction ----------------

def probe_tracks(path: str | Path) -> list[dict]:
    """All audio streams in the file: index (within audio streams), codec, title tag."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         "-select_streams", "a", str(path)], stderr=subprocess.DEVNULL)
    streams = json.loads(out).get("streams", [])
    tracks = []
    for i, s in enumerate(streams):
        tags = s.get("tags") or {}
        tracks.append({
            "index": i,
            "codec": s.get("codec_name"),
            "channels": s.get("channels"),
            "sample_rate": s.get("sample_rate"),
            "title": tags.get("title") or tags.get("handler_name") or None,
            "language": tags.get("language"),
        })
    return tracks


def extract_track(path: str | Path, track_index: int, out_wav: Path) -> Path:
    """One audio stream -> 16 kHz mono PCM WAV (the format both engines consume)."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path),
                    "-map", f"0:a:{track_index}", "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(out_wav)], check=True)
    if not out_wav.exists() or out_wav.stat().st_size <= 44:
        raise RuntimeError(f"ffmpeg extracted no audio from track {track_index} of {path}")
    return out_wav


def wav_duration(path: str | Path) -> float:
    import wave
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ---------------- duplicate-track detection ----------------

def _wav_window(path: str | Path, start_s: float, dur_s: float) -> list[float]:
    """PCM samples of one window as floats (16 kHz mono s16le wav from extract_track)."""
    import wave
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        w.setpos(min(int(start_s * sr), w.getnframes()))
        raw = w.readframes(int(dur_s * sr))
    n = len(raw) // 2
    return list(struct.unpack(f"<{n}h", raw[:n * 2]))


def _pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 100:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    cov = va = vb = 0.0
    for x, y in zip(a, b):
        dx, dy = x - ma, y - mb
        cov += dx * dy
        va += dx * dx
        vb += dy * dy
    if va <= 0 or vb <= 0:
        return 1.0 if va == vb else 0.0  # two silent windows are "identical"
    return cov / (va ** 0.5 * vb ** 0.5)


def tracks_duplicate(wav_a: Path, wav_b: Path) -> bool:
    """True when two extracted tracks carry the same audio (mixdown copies).

    Zero-lag correlation on three sampled windows: real per-participant tracks
    contain different voices, so their correlation is low even when they overlap."""
    dur = min(wav_duration(wav_a), wav_duration(wav_b))
    if dur <= 0:
        return True
    win = min(5.0, dur)
    scores = []
    for frac in (0.1, 0.5, 0.85):
        start = max(0.0, min(dur - win, dur * frac))
        scores.append(_pearson(_wav_window(wav_a, start, win),
                               _wav_window(wav_b, start, win)))
    return min(scores) > 0.9


# ---------------- engine clients (localhost containers) ----------------

def asr_transcribe(port: int, wav: Path, timeout: float) -> dict:
    try:
        with open(wav, "rb") as f:
            r = httpx.post(f"http://127.0.0.1:{port}/v1/audio/transcriptions",
                           files={"file": (wav.name, f, "audio/wav")},
                           data={"timestamps": "true"}, timeout=timeout)
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"ASR engine connection failed mid-transcription ({e}); the container "
            "may have been killed (memory cap?) - check `aisee model logs` and "
            "`docker inspect` OOMKilled") from e
    if r.status_code != 200:
        raise RuntimeError(f"ASR engine HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def diarize(port: int, wav: Path, timeout: float, min_speakers: int | None = None,
            max_speakers: int | None = None, num_speakers: int | None = None) -> dict:
    data = {}
    if min_speakers:
        data["min_speakers"] = str(min_speakers)
    if max_speakers:
        data["max_speakers"] = str(max_speakers)
    if num_speakers:
        data["num_speakers"] = str(num_speakers)
    with open(wav, "rb") as f:
        r = httpx.post(f"http://127.0.0.1:{port}/v1/audio/diarizations",
                       files={"file": (wav.name, f, "audio/wav")}, data=data,
                       timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"diarization engine HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


# ---------------- word->speaker merge + segment building ----------------

def merge_sliver_speakers(turns: list[dict], min_talk_s: float = 5.0
                          ) -> tuple[list[dict], int]:
    """pyannote over-splits: speakers with almost no total talk are cluster shards,
    not people. Reassign their turns to the temporally nearest real speaker (a real
    participant says at least a few sentences). Returns (turns, merged_count)."""
    talk: dict[str, float] = {}
    for t in turns:
        talk[t["speaker"]] = talk.get(t["speaker"], 0.0) + t["end"] - t["start"]
    keep = {s for s, v in talk.items() if v >= min_talk_s}
    if not keep or len(keep) == len(talk):
        return turns, 0
    real = sorted((t for t in turns if t["speaker"] in keep), key=lambda t: t["start"])
    out = []
    for t in turns:
        if t["speaker"] in keep:
            out.append(t)
            continue
        mid = (t["start"] + t["end"]) / 2
        nearest = min(real, key=lambda r: min(abs(r["start"] - mid), abs(r["end"] - mid)))
        out.append({**t, "speaker": nearest["speaker"]})
    return sorted(out, key=lambda t: t["start"]), len(talk) - len(keep)


def assign_speakers(words: list[dict], turns: list[dict]) -> list[dict]:
    """Label each ASR word with the diarization turn covering its midpoint
    (nearest turn within 1 s when none covers it). Turns must be time-sorted."""
    turns = sorted(turns, key=lambda t: t["start"])
    out = []
    ti = 0
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        while ti + 1 < len(turns) and turns[ti]["end"] < mid \
                and turns[ti + 1]["start"] <= mid:
            ti += 1
        best, best_d = None, 1.0
        for t in turns[max(0, ti - 1):ti + 3]:
            if t["start"] <= mid <= t["end"]:
                best, best_d = t, 0.0
                break
            d = min(abs(t["start"] - mid), abs(t["end"] - mid))
            if d < best_d:
                best, best_d = t, d
        out.append({**w, "speaker": best["speaker"] if best else None})
    return out


def words_to_segments(words: list[dict], gap_s: float = 1.5,
                      max_len_s: float = 30.0) -> list[dict]:
    """Group speaker-labeled words into display segments: split on speaker change,
    silence gaps, or an overlong run."""
    segs: list[dict] = []
    for w in words:
        s = segs[-1] if segs else None
        if (s and s["speaker"] == w.get("speaker")
                and w["start"] - s["end"] <= gap_s
                and w["end"] - s["start"] <= max_len_s):
            s["end"] = w["end"]
            s["text"] += " " + w["word"]
        else:
            segs.append({"start": w["start"], "end": w["end"],
                         "speaker": w.get("speaker"), "text": w["word"]})
    for s in segs:
        s["start"], s["end"] = round(s["start"], 3), round(s["end"], 3)
    return segs


def merge_track_words(per_track: list[tuple[str, list[dict]]]) -> list[dict]:
    """Interleave per-track word streams by time; each track is one speaker."""
    words = []
    for speaker, ws in per_track:
        for w in ws:
            words.append({**w, "speaker": speaker})
    return sorted(words, key=lambda w: (w["start"], w["end"]))


# ---------------- artifacts ----------------

def _fmt_ts(t: float, vtt: bool = False) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    sep = "." if vtt else ","
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int((s % 1) * 1000):03d}"


def write_artifacts(out_dir: Path, result: dict, words: list[dict]) -> list[str]:
    """transcript.json (full: result + words), .txt (speaker-labeled), .srt, .vtt.
    Returns the artifact filenames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    segs = result.get("segments") or []
    (out_dir / "transcript.json").write_text(
        json.dumps({**result, "words": words}, indent=1))
    lines = []
    for s in segs:
        who = f"{s['speaker']}: " if s.get("speaker") else ""
        lines.append(f"[{_fmt_ts(s['start'], vtt=True)}] {who}{s['text']}")
    (out_dir / "transcript.txt").write_text("\n".join(lines) + "\n")
    srt, vtt = [], ["WEBVTT", ""]
    for i, s in enumerate(segs, 1):
        who = f"{s['speaker']}: " if s.get("speaker") else ""
        srt += [str(i), f"{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}",
                who + s["text"], ""]
        vtt += [f"{_fmt_ts(s['start'], vtt=True)} --> {_fmt_ts(s['end'], vtt=True)}",
                who + s["text"], ""]
    (out_dir / "transcript.srt").write_text("\n".join(srt))
    (out_dir / "transcript.vtt").write_text("\n".join(vtt))
    return ["transcript.json", "transcript.txt", "transcript.srt", "transcript.vtt"]


def write_diarization_artifacts(out_dir: Path, result: dict, uri: str) -> list[str]:
    """diarization.json + standard RTTM."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diarization.json").write_text(json.dumps(result, indent=1))
    rttm = [f"SPEAKER {uri} 1 {t['start']:.3f} {t['end'] - t['start']:.3f} "
            f"<NA> <NA> {t['speaker']} <NA> <NA>" for t in result.get("turns", [])]
    (out_dir / "diarization.rttm").write_text("\n".join(rttm) + "\n")
    return ["diarization.json", "diarization.rttm"]
