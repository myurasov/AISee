# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""ASR serving app (NeMo Parakeet family) - runs inside the aisee audio-nemo image.

One resident model, one inference at a time (GPU discipline: unified-memory hosts
OOM the OS under concurrent heavy jobs). Startup is GPU-gated: the model must sit
on CUDA and transcribe a 10 s probe fast, else the process exits nonzero so the
host's wait_ready surfaces the failure loudly instead of a silent CPU fallback.
"""

import argparse
import math
import os
import struct
import sys
import tempfile
import threading
import time
import wave

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
args = ap.parse_args()

import torch  # noqa: E402  (container torch, sm_121-native)

if not torch.cuda.is_available():
    print("FATAL: CUDA not available in the serving container", flush=True)
    sys.exit(3)

import nemo.collections.asr as nemo_asr  # noqa: E402

print(f"loading {args.model} ...", flush=True)
t0 = time.time()
MODEL = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
MODEL = MODEL.cuda().eval()
print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

# model card: full attention is reliable up to ~24 min of audio; switch to local
# attention beyond that (and back, so short inputs keep full-attention quality)
LONG_AUDIO_S = 1200.0
_attn_mode = "full"
_lock = threading.Lock()


def _set_attention(mode: str) -> None:
    global _attn_mode
    if mode == _attn_mode:
        return
    if mode == "local":
        MODEL.change_attention_model(self_attention_model="rel_pos_local_attn",
                                     att_context_size=[256, 256])
    else:
        MODEL.change_attention_model(self_attention_model="rel_pos",
                                     att_context_size=[-1, -1])
    _attn_mode = mode
    print(f"attention mode -> {mode}", flush=True)


def _wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _write_probe_wav(path: str, seconds: float = 10.0, sr: int = 16000) -> None:
    """Synthetic speech-band probe (chirpy tone bursts) - content does not matter,
    only that the encoder+decoder run end to end on the GPU."""
    n = int(seconds * sr)
    frames = bytearray()
    for i in range(n):
        t = i / sr
        env = 1.0 if (t % 1.0) < 0.6 else 0.05
        v = 0.3 * env * math.sin(2 * math.pi * (180 + 90 * math.sin(t * 2)) * t)
        frames += struct.pack("<h", int(v * 32767))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))


def _transcribe(path: str, timestamps: bool = True) -> dict:
    dur = _wav_duration(path)
    with _lock:
        _set_attention("local" if dur > LONG_AUDIO_S else "full")
        t0 = time.time()
        out = MODEL.transcribe([path], timestamps=timestamps)
        wall = time.time() - t0
    r = out[0]
    words, segments = [], []
    ts = getattr(r, "timestamp", None) or {}
    for w in ts.get("word", []) or []:
        words.append({"word": w.get("word", ""), "start": round(float(w["start"]), 3),
                      "end": round(float(w["end"]), 3)})
    for s in ts.get("segment", []) or []:
        segments.append({"start": round(float(s["start"]), 3),
                         "end": round(float(s["end"]), 3),
                         "text": s.get("segment", s.get("text", ""))})
    return {"text": r.text or "", "words": words, "segments": segments,
            "duration_s": round(dur, 2), "wall_s": round(wall, 2),
            "rtfx": round(dur / wall, 1) if wall > 0 else None,
            "attention": _attn_mode, "model": args.model}


# ---- GPU gate: prove CUDA inference before serving ----
_probe_rtfx = None
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    _write_probe_wav(f.name)
    try:
        p = next(MODEL.parameters())
        if not p.is_cuda:
            print("FATAL: model parameters are not on CUDA", flush=True)
            sys.exit(3)
        _transcribe(f.name, timestamps=True)  # warmup: CUDA context + autotune
        gate = _transcribe(f.name, timestamps=True)
    finally:
        os.unlink(f.name)
if gate["wall_s"] > 5.0:
    print(f"FATAL: GPU gate failed - 10 s probe took {gate['wall_s']}s "
          "(silent CPU fallback?)", flush=True)
    sys.exit(3)
_probe_rtfx = gate["rtfx"]
print(f"GPU gate ok: probe rtfx {_probe_rtfx}", flush=True)

from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402

app = FastAPI(title="aisee-audio-nemo")


@app.get("/health")
def health():
    return {"ok": True, "engine": "nemo-asr", "model": args.model,
            "device": "cuda", "probe_rtfx": _probe_rtfx, "attention": _attn_mode}


@app.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...), timestamps: bool = Form(True)):
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        return _transcribe(path, timestamps=timestamps)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
