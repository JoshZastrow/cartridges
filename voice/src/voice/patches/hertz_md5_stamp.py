"""Hash hertz-dev's checkpoints once; a stamp file next to each skips the MD5 on later starts.

Upstream's `utils.dist.load_ckpt` MD5s each checkpoint on every start (the
whole file read into memory, then hashed), before `torch.load` reads it again.
That is most of a 352 s cold start from the `voice-data` Volume (#21, #27).
The patch replaces `load_ckpt` in upstream's modules with one that:

- downloads (or finds in the HF cache) the file as upstream does;
- trusts a stamp `<file>.md5stamp` recording {path, size, mtime_ns, md5} while
  the file's path, size and mtime_ns still match and the md5 is the expected one;
- otherwise hashes the file (streamed, in 64 MiB chunks) and, on a match,
  writes the stamp. A missing or garbled stamp means hash again;
- on a mismatch deletes the file and downloads it once more, as upstream does;
  a second mismatch raises. Unlike upstream it also deletes the blob the cache
  symlink points at: removing only the symlink lets `hf_hub_download` relink
  the same bad blob, and upstream's retry loops forever. It refuses to delete
  a blob whose realpath is not under `HF_HUB_CACHE`.

The MD5 is an integrity check against truncated or corrupted downloads, not a
security control. The stamp keeps that check for any change that touches the
file's size or mtime. It does not catch bit-rot that leaves both unchanged.

Single process: upstream's `torch.distributed` broadcast of the path is not
kept. The caller commits the Volume after load so new stamps persist
(`voice.deploy_hertz.loaded_backend`).

    apply()   # after upstream is on sys.path (HertzBackend.load does this)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

REPO = "si-pbc/hertz-dev"  # upstream load_ckpt hard-codes this repo
SUFFIX = ".md5stamp"
READ = 64 << 20  # bytes per read while hashing
MODULES = ("utils.dist", "utils", "model", "tokenizer", "transformer")  # upstream modules that bind load_ckpt


def stamp_path(path: str | Path) -> Path:
    return Path(f"{path}{SUFFIX}")


def md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(READ):
            h.update(block)
    return h.hexdigest()


def stamp(path: str | Path, digest: str) -> dict:
    st = os.stat(path)
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "md5": digest}


def stamped(path: str | Path, expected: str) -> bool:
    """True when the stamp next to `path` records this file as it is now, with md5 `expected`."""
    try:
        return json.loads(stamp_path(path).read_text()) == stamp(path, expected)
    except (OSError, ValueError):
        return False


def verify(path: str | Path, expected: str) -> bool:
    """True when `path` has md5 `expected`: from the stamp if it still matches,
    else by hashing the file (and stamping it on a match)."""
    if stamped(path, expected):
        return True
    if md5(path) != expected:
        return False
    stamp_path(path).write_text(json.dumps(stamp(path, expected)))
    return True


def download(name: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=REPO, filename=f"{name}.pt")


def remove(path: str) -> None:
    """The cache symlink, the blob it points at, and the stamp. Refuses a blob outside `HF_HUB_CACHE`."""
    from huggingface_hub import constants

    blob, cache = os.path.realpath(path), os.path.realpath(constants.HF_HUB_CACHE)
    if os.path.commonpath([blob, cache]) != cache:
        raise RuntimeError(f"refusing to delete {blob}: not under HF_HUB_CACHE {cache}")
    for p in {blob, path, str(stamp_path(path))}:
        if os.path.lexists(p):
            os.remove(p)


def load_ckpt(load_from_location: str, expected_hash: str | None = None):
    """Upstream `utils.dist.load_ckpt` for one process, with `verify` in place of its MD5."""
    import torch

    path = download(load_from_location)
    if expected_hash is not None and not verify(path, expected_hash):
        logger.warning("MD5 mismatch for %s (expected %s); deleting it and downloading again", path, expected_hash)
        remove(path)
        path = download(load_from_location)
        if not verify(path, expected_hash):
            raise RuntimeError(f"MD5 mismatch for {path} after a fresh download (expected {expected_hash})")
    return torch.load(path, weights_only=False, map_location="cpu")


def apply() -> None:
    import importlib

    for name in MODULES:
        importlib.import_module(name).load_ckpt = load_ckpt
