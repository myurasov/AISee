# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Audio pipeline: lane probing/extraction (audio files and video containers),
ASR/diarization engine clients, word->speaker merge, and transcript artifacts.

Design notes (see res/add-audio-implementation-report.md in the project):
- AISee does NOT interpret what tracks/channels mean (author mic, system audio,
  per-participant, ...). Every audio stream - and every CHANNEL of a multi-channel
  stream, since stereo is often two encoded tracks - is one independent mono
  "lane"; each lane gets its own transcript (and diarization when asked). Combining
  lanes is the consumer's job. The only cross-lane logic is factual: BIT-IDENTICAL
  lanes (dual-mono channels, copied tracks - equal PCM hashes after extraction) are
  not transcribed twice; merely similar lanes are always processed independently.
- Serving containers receive 16 kHz mono WAV over multipart HTTP (localhost), so
  they never need access to host paths or the task media dir.
- Segment/word timestamps are absolute positions in the recording.
"""

import json
import subprocess
from pathlib import Path

import httpx


# ---------------- probing + extraction ----------------

def probe_lanes(path: str | Path) -> list[dict]:
    """Independent mono lanes of the file: one per (audio stream, channel).

    A mono stream is one lane; a C-channel stream is C lanes (stereo often encodes
    two tracks). Lane labels come from the track title tag when present, else
    audio<i>, with -L/-R (or -c<k>) suffixes for split channels."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         "-select_streams", "a", str(path)], stderr=subprocess.DEVNULL)
    streams = json.loads(out).get("streams", [])
    lanes = []
    for i, s in enumerate(streams):
        tags = s.get("tags") or {}
        title = tags.get("title") or None
        base = (title or f"audio{i}").strip()
        ch = int(s.get("channels") or 1)
        for k in range(max(ch, 1)):
            if ch <= 1:
                label = base
            elif ch == 2:
                label = f"{base}-{'LR'[k]}"
            else:
                label = f"{base}-c{k}"
            lanes.append({
                "label": label,
                "stream": i,
                "channel": k if ch > 1 else None,
                "stream_channels": ch,
                "codec": s.get("codec_name"),
                "title": title,
                "language": tags.get("language"),
            })
    return lanes


def extract_lane(path: str | Path, lane: dict, out_wav: Path) -> Path:
    """One lane -> 16 kHz mono PCM WAV (the format both engines consume)."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(path),
           "-map", f"0:a:{lane['stream']}", "-vn"]
    if lane.get("channel") is not None:
        # pick one channel of a multi-channel stream instead of downmixing
        cmd += ["-af", f"pan=mono|c0=c{lane['channel']}"]
    else:
        cmd += ["-ac", "1"]
    cmd += ["-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)]
    subprocess.run(cmd, check=True)
    if not out_wav.exists() or out_wav.stat().st_size <= 44:
        raise RuntimeError(f"ffmpeg extracted no audio for lane {lane['label']} of {path}")
    return out_wav


def wav_duration(path: str | Path) -> float:
    import wave
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ---------------- duplicate-lane detection ----------------

def pcm_sha256(wav: str | Path) -> str:
    """SHA-256 of the decoded PCM frames (WAV header excluded).

    Lanes extracted through the same deterministic decode/resample pipeline hash
    equal exactly when the source audio is bit-identical (dual-mono channels, copied
    tracks) - so dedup skips work ONLY on provable duplicates, never on merely
    similar lanes (verified: correlated-but-distinct lanes hash differently)."""
    import hashlib
    import wave
    h = hashlib.sha256()
    with wave.open(str(wav), "rb") as w:
        while True:
            chunk = w.readframes(1 << 16)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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


# ---------------- artifacts ----------------

def _fmt_ts(t: float, vtt: bool = False) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    sep = "." if vtt else ","
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int((s % 1) * 1000):03d}"


def safe_label(label: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)[:60] or "lane"


def _write_rendered(out_dir: Path, suffix: str, segs: list[dict]) -> list[str]:
    """transcript<suffix>.txt/.srt/.vtt for one lane's segments."""
    lines = []
    for s in segs:
        who = f"{s['speaker']}: " if s.get("speaker") else ""
        lines.append(f"[{_fmt_ts(s['start'], vtt=True)}] {who}{s['text']}")
    (out_dir / f"transcript{suffix}.txt").write_text("\n".join(lines) + "\n")
    srt, vtt = [], ["WEBVTT", ""]
    for i, s in enumerate(segs, 1):
        who = f"{s['speaker']}: " if s.get("speaker") else ""
        srt += [str(i), f"{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}",
                who + s["text"], ""]
        vtt += [f"{_fmt_ts(s['start'], vtt=True)} --> {_fmt_ts(s['end'], vtt=True)}",
                who + s["text"], ""]
    (out_dir / f"transcript{suffix}.srt").write_text("\n".join(srt))
    (out_dir / f"transcript{suffix}.vtt").write_text("\n".join(vtt))
    return [f"transcript{suffix}.txt", f"transcript{suffix}.srt", f"transcript{suffix}.vtt"]


def write_artifacts(out_dir: Path, result: dict,
                    words_by_lane: dict[str, list[dict]]) -> list[str]:
    """transcript.json (full result + per-lane words) plus rendered txt/srt/vtt.
    Single lane keeps the plain names; multiple lanes get transcript-<label>.*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    single = len(words_by_lane) == 1
    (out_dir / "transcript.json").write_text(json.dumps(
        {**result, "words": (next(iter(words_by_lane.values())) if single
                             else words_by_lane)}, indent=1))
    names = ["transcript.json"]
    lanes = result.get("tracks") or []
    for lane in lanes:
        if lane.get("duplicate_of"):
            continue
        suffix = "" if single else f"-{safe_label(lane['label'])}"
        names += _write_rendered(out_dir, suffix, lane.get("segments") or [])
    return names


def write_diarization_artifacts(out_dir: Path, result: dict, uri: str) -> list[str]:
    """diarization.json + standard RTTM (per lane when there are several)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diarization.json").write_text(json.dumps(result, indent=1))
    names = ["diarization.json"]
    lanes = result.get("tracks") or []
    single = len([ln for ln in lanes if not ln.get("duplicate_of")]) == 1
    for lane in lanes:
        if lane.get("duplicate_of"):
            continue
        suffix = "" if single else f"-{safe_label(lane['label'])}"
        rttm = [f"SPEAKER {uri} 1 {t['start']:.3f} {t['end'] - t['start']:.3f} "
                f"<NA> <NA> {t['speaker']} <NA> <NA>" for t in lane.get("turns") or []]
        (out_dir / f"diarization{suffix}.rttm").write_text("\n".join(rttm) + "\n")
        names.append(f"diarization{suffix}.rttm")
    return names
