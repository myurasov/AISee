# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed media store: skip re-uploading bytes the server already has.

Blobs live in tasks/blobs/ as <sha256><ext> (the extension is kept because the media
pipeline sniffs image-vs-video by suffix). Tasks hardlink blobs into their own media dir,
so GC can unlink old blobs freely - the tasks' hardlinked copies share the inode and
survive. Reuse touches the blob's mtime, which is what "recently" means for GC.
"""

import hashlib
import os
import time
from pathlib import Path

from . import paths


def blobs_dir() -> Path:
    return paths.tasks_dir() / "blobs"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find(sha: str) -> Path | None:
    """The stored blob for a hash, or None. Touches mtime so reuse extends retention."""
    if not sha or len(sha) != 64 or not all(c in "0123456789abcdef" for c in sha.lower()):
        return None
    for p in blobs_dir().glob(sha.lower() + "*"):
        os.utime(p, None)
        return p
    return None


def put_bytes(data: bytes, filename: str) -> tuple[str, Path]:
    """Store bytes as a blob (no-op if already present); returns (sha256, blob path)."""
    sha = hashlib.sha256(data).hexdigest()
    existing = find(sha)
    if existing:
        return sha, existing
    d = blobs_dir()
    d.mkdir(parents=True, exist_ok=True)
    ext = Path(os.path.basename(filename)).suffix.lower()[:16]
    p = d / (sha + ext)
    tmp = d / (sha + ext + ".tmp")
    tmp.write_bytes(data)
    tmp.rename(p)
    return sha, p


def link_into(blob: Path, dest_dir: Path, filename: str | None = None) -> Path:
    """Hardlink a blob into a task's staging dir (copy if the fs refuses links)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (os.path.basename(filename) if filename else blob.name)
    if dest.exists():
        return dest
    try:
        os.link(blob, dest)
    except OSError:
        dest.write_bytes(blob.read_bytes())
    return dest


def gc(ttl_hours: float) -> int:
    """Unlink blobs not stored/reused within the TTL window (0 disables GC)."""
    if not ttl_hours:
        return 0
    cutoff = time.time() - ttl_hours * 3600
    n = 0
    d = blobs_dir()
    if not d.is_dir():
        return 0
    for p in d.iterdir():
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                n += 1
        except OSError:
            pass
    return n
