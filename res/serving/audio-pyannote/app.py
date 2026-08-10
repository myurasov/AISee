# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Diarization serving app (pyannote.audio 3.x) - runs inside the aisee
audio-pyannote image.

Carries the three verified GB10/ARM64 patches (see the AISee audio report):
capability spoof for NVRTC-JIT paths, torch.load weights_only=False for the 3.1
checkpoint, and a probe-and-patch replacement of complex-CUDA .abs() (cu128 nvrtc
cannot JIT for sm_121). Audio goes in as an in-memory waveform, bypassing
pyannote's torchaudio file I/O. Startup is GPU-gated: unpatched pyannote silently
runs on CPU ~120x slower - the worst failure mode - so a slow probe kills the
process instead of serving degraded.
"""

import argparse
import gc
import math
import os
import struct
import sys
import tempfile
import threading
import time
import wave
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="pyannote/speaker-diarization-3.1")
args = ap.parse_args()

import torch  # noqa: E402

if not torch.cuda.is_available():
    print("FATAL: CUDA not available in the serving container", flush=True)
    sys.exit(3)

# GB10 patch 1: report SM 9.0 so NVRTC-JIT'd python-side kernels compile
torch.cuda.get_device_capability = lambda *a, **k: (9, 0)

# torchaudio >=2.9 removed the legacy I/O surface (AudioMetaData, list_audio_backends,
# info, load); pyannote 3.x imports all of it at module load. Restore it backed by
# soundfile (verified shim from the GB10 lab; handles every 16 kHz mono wav we send).
import torchaudio  # noqa: E402

if not hasattr(torchaudio, "AudioMetaData"):
    class _AudioMetaData:
        def __init__(self, sample_rate=0, num_frames=0, num_channels=0,
                     bits_per_sample=0, encoding=""):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    torchaudio.AudioMetaData = _AudioMetaData
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *a, **k: None
if not hasattr(torchaudio, "info"):
    def _ta_info(filepath, *a, **k):
        import soundfile as _sf
        i = _sf.info(str(filepath))
        return torchaudio.AudioMetaData(sample_rate=int(i.samplerate),
                                        num_frames=int(i.frames),
                                        num_channels=int(i.channels),
                                        bits_per_sample=16, encoding="PCM_S")
    torchaudio.info = _ta_info
if not hasattr(torchaudio, "load"):
    def _ta_load(filepath, frame_offset=0, num_frames=-1, *a, **k):
        import soundfile as _sf
        data, sr = _sf.read(str(filepath), start=frame_offset,
                            frames=num_frames if num_frames > 0 else -1,
                            dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sr
    torchaudio.load = _ta_load

# GB10 patch 2: torch>=2.6 defaults weights_only=True; the official 3.1
# checkpoint (trusted source) trips it
_orig_load = torch.load


def _load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)


torch.load = _load

import soundfile as sf  # noqa: E402
from pyannote.audio import Pipeline  # noqa: E402

print(f"loading {args.model} ...", flush=True)
t0 = time.time()
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
PIPE = Pipeline.from_pretrained(args.model, use_auth_token=token)
if PIPE is None:
    print(f"FATAL: could not load {args.model} - is the HF token authorized for "
          "the gated pyannote repos (3.1 + segmentation-3.0 + wespeaker)?", flush=True)
    sys.exit(3)
PIPE.to(torch.device("cuda"))
print(f"pipeline loaded in {time.time() - t0:.1f}s", flush=True)

# GB10 patch 3: probe complex-CUDA .abs(); if nvrtc cannot JIT it for this arch,
# compute the magnitude manually and keep everything else on GPU eager kernels
try:
    torch.fft.rfft(torch.randn(8, device="cuda")).abs()
except RuntimeError as e:
    if "nvrtc" in str(e) or "gpu-architecture" in str(e):
        _orig_abs = torch.Tensor.abs

        def _safe_abs(self):
            if self.is_cuda and self.is_complex():
                return torch.sqrt(self.real * self.real + self.imag * self.imag)
            return _orig_abs(self)

        torch.Tensor.abs = _safe_abs
        print("patched complex-CUDA Tensor.abs (nvrtc lacks this arch)", flush=True)
    else:
        raise

_lock = threading.Lock()


def _diarize(path: str, min_speakers: int | None = None,
             max_speakers: int | None = None, num_speakers: int | None = None) -> dict:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    dur = len(data) / sr
    waveform = torch.from_numpy(data).unsqueeze(0)
    kw = {}
    if num_speakers:
        kw["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kw["min_speakers"] = min_speakers
        if max_speakers:
            kw["max_speakers"] = max_speakers
    with _lock:
        t0 = time.time()
        dia = PIPE({"waveform": waveform, "sample_rate": sr}, **kw)
        wall = time.time() - t0
    turns = [{"start": round(seg.start, 3), "end": round(seg.end, 3), "speaker": label}
             for seg, _, label in dia.itertracks(yield_label=True)]
    talk = Counter()
    for t in turns:
        talk[t["speaker"]] += t["end"] - t["start"]
    # unified memory: drop references, collect, then trim so blocks return to the system
    del dia, waveform, data
    gc.collect()
    torch.cuda.empty_cache()
    return {"turns": turns, "num_speakers": len(talk),
            "speakers": {k: round(v, 1) for k, v in talk.items()},
            "duration_s": round(dur, 2), "wall_s": round(wall, 2),
            "rtfx": round(dur / wall, 1) if wall > 0 else None, "model": args.model}


def _write_probe_wav(path: str, seconds: float = 10.0, sr: int = 16000) -> None:
    n = int(seconds * sr)
    frames = bytearray()
    for i in range(n):
        t = i / sr
        f0 = 140 if t < seconds / 2 else 220  # two "voices"
        env = 1.0 if (t % 0.8) < 0.5 else 0.02
        v = 0.3 * env * math.sin(2 * math.pi * f0 * t) + 0.02 * math.sin(2 * math.pi * 1700 * t)
        frames += struct.pack("<h", int(v * 32767))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))


# ---- GPU gate ----
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    _write_probe_wav(f.name)
try:
    _diarize(f.name)  # warmup: CUDA context + autotune outside the timed gate
    gate = _diarize(f.name)
finally:
    os.unlink(f.name)
if gate["wall_s"] > 10.0:
    print(f"FATAL: GPU gate failed - 10 s probe took {gate['wall_s']}s "
          "(pyannote silently on CPU?)", flush=True)
    sys.exit(3)
print(f"GPU gate ok: probe rtfx {gate['rtfx']}", flush=True)

from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402

app = FastAPI(title="aisee-audio-pyannote")


@app.get("/health")
def health():
    return {"ok": True, "engine": "pyannote", "model": args.model, "device": "cuda",
            "probe_rtfx": gate["rtfx"]}


@app.post("/v1/audio/diarizations")
async def diarizations(file: UploadFile = File(...),
                       min_speakers: int | None = Form(None),
                       max_speakers: int | None = Form(None),
                       num_speakers: int | None = Form(None)):
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        return _diarize(path, min_speakers=min_speakers, max_speakers=max_speakers,
                        num_speakers=num_speakers)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
