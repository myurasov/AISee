# AISee diarization serving image (pyannote.audio 3.x).
# Plain python base + PyPI torch 2.11+cu128: the only PyPI torch line whose aarch64
# wheels import cleanly on GB10-class hosts (2.7/2.9 are broken at import). The GPU
# runtime libs come from the torch wheel + the driver injected by the NVIDIA toolkit.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# torch/torchaudio resolved from the cu128 index (never bare PyPI - broken wheels);
# huggingface_hub<1.0 keeps pyannote's use_auth_token kwarg alive (removed in 1.0)
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        "torch==2.11.*" "torchaudio==2.11.*" && \
    pip install --no-cache-dir "pyannote.audio>=3.1,<4" "huggingface_hub<1.0" \
        soundfile fastapi uvicorn python-multipart

COPY app.py /app/app.py

ENV HF_HOME=/hf-cache
