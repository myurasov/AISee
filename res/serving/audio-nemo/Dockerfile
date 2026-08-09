# AISee audio ASR serving image (NeMo / Parakeet family).
# Base: NGC PyTorch - its torch (2.10+nv) JITs sm_121 natively on GB10-class GPUs;
# never replace it with a PyPI torch here (broken aarch64 wheels, no sm_121 NVRTC).
FROM nvcr.io/nvidia/pytorch:25.11-py3

# lhotse pin per verified GB10 recipe; nemo picks up the container torch (already
# satisfied, so pip does not touch it - the build assert below guards that)
RUN pip install --no-cache-dir "nemo_toolkit[asr]==2.5.3" "lhotse>=1.24,<1.30" \
        fastapi uvicorn python-multipart && \
    python -c "import torch; v = torch.__version__; assert 'nv' in v or 'a0' in v, f'container torch was replaced: {v}'"

COPY app.py /app/app.py

ENV HF_HOME=/hf-cache
# port and model come from the container command (aisee dockerctl builds it)
