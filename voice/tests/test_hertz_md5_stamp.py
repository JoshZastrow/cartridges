"""patches/hertz_md5_stamp.py: a matching stamp skips the hash; a size or mtime
change, or a missing or garbled stamp, re-hashes; a corrupted file still fails.
CPU only: `download` and `md5` are monkeypatched, nothing reaches HF."""

import hashlib
import os
import sys
import types

import pytest
import torch
from voice.patches import hertz_md5_stamp as patch


@pytest.fixture
def ckpt(tmp_path, monkeypatch):
    """A checkpoint in an HF-cache-like layout under HF_HUB_CACHE: snapshot symlink -> blob."""
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    blob = tmp_path / "blobs" / "abc"
    blob.parent.mkdir()
    torch.save({"w": torch.arange(4)}, blob)
    link = tmp_path / "snapshots" / "rev" / "inference_x.pt"
    link.parent.mkdir(parents=True)
    link.symlink_to(blob)
    return link, hashlib.md5(blob.read_bytes()).hexdigest()


@pytest.fixture
def hashes(monkeypatch):
    """Counts calls to the real md5."""
    calls = []
    real = patch.md5

    def counted(path):
        calls.append(str(path))
        return real(path)

    monkeypatch.setattr(patch, "md5", counted)
    return calls


def test_first_verify_hashes_and_writes_the_stamp(ckpt, hashes):
    path, digest = ckpt
    assert patch.verify(path, digest)
    assert len(hashes) == 1
    assert patch.stamped(path, digest)


def test_matching_stamp_skips_the_hash(ckpt, hashes):
    path, digest = ckpt
    patch.verify(path, digest)
    assert patch.verify(path, digest) and patch.verify(path, digest)
    assert len(hashes) == 1


def test_stamp_for_another_expected_hash_does_not_match(ckpt, hashes):
    path, digest = ckpt
    patch.verify(path, digest)
    assert not patch.verify(path, "0" * 32)
    assert len(hashes) == 2


def test_mtime_change_rehashes(ckpt, hashes):
    path, digest = ckpt
    patch.verify(path, digest)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert patch.verify(path, digest)
    assert len(hashes) == 2
    assert patch.stamped(path, digest)  # re-stamped with the new mtime


def test_size_change_rehashes_and_fails(ckpt, hashes):
    path, digest = ckpt
    patch.verify(path, digest)
    st = os.stat(path)
    with open(path, "ab") as f:
        f.write(b"\0")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # same mtime: the size alone forces the hash
    assert not patch.verify(path, digest)
    assert len(hashes) == 2


@pytest.mark.parametrize("content", [None, "", "{not json", '{"path": "x"}', "[]"])
def test_missing_or_garbled_stamp_rehashes(ckpt, hashes, content):
    path, digest = ckpt
    patch.verify(path, digest)
    stamp = patch.stamp_path(path)
    stamp.unlink()
    if content is not None:
        stamp.write_text(content)
    assert patch.verify(path, digest)
    assert len(hashes) == 2


def test_load_ckpt_with_a_stamp_loads_without_hashing(ckpt, hashes, monkeypatch):
    path, digest = ckpt
    monkeypatch.setattr(patch, "download", lambda name: str(path))
    patch.verify(path, digest)
    assert torch.equal(patch.load_ckpt("inference_x", digest)["w"], torch.arange(4))
    assert len(hashes) == 1


def test_load_ckpt_without_expected_hash_never_hashes(ckpt, hashes, monkeypatch):
    path, _ = ckpt
    monkeypatch.setattr(patch, "download", lambda name: str(path))
    patch.load_ckpt("inference_x")
    assert hashes == [] and not patch.stamp_path(path).exists()


def test_corrupted_file_deletes_redownloads_then_fails(ckpt, monkeypatch):
    path, digest = ckpt
    blob = path.resolve()
    downloads = []

    def download(name):  # the hub serves the same corrupted bytes again
        downloads.append(blob.exists())
        if not blob.exists():
            blob.write_bytes(b"corrupt")
        if not path.is_symlink():
            path.symlink_to(blob)
        return str(path)

    blob.write_bytes(b"corrupt")
    monkeypatch.setattr(patch, "download", download)
    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        patch.load_ckpt("inference_x", digest)
    assert downloads == [True, False]  # the bad blob was deleted before the second download
    assert not patch.stamp_path(path).exists()


def test_corrupted_file_recovers_when_the_redownload_is_good(ckpt, monkeypatch):
    path, digest = ckpt
    blob = path.resolve()
    good = blob.read_bytes()
    blob.write_bytes(b"corrupt")

    def download(name):
        if not blob.exists():  # the bad blob was deleted, so the hub fetches it again
            blob.write_bytes(good)
        if not path.is_symlink():
            path.symlink_to(blob)
        return str(path)

    monkeypatch.setattr(patch, "download", download)
    assert torch.equal(patch.load_ckpt("inference_x", digest)["w"], torch.arange(4))
    assert patch.stamped(path, digest)


def test_remove_refuses_a_blob_outside_the_cache(ckpt, tmp_path, monkeypatch):
    path, _ = ckpt
    blob = path.resolve()
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "snapshots"))
    with pytest.raises(RuntimeError, match="not under HF_HUB_CACHE"):
        patch.remove(str(path))
    assert blob.exists() and path.is_symlink()


def test_apply_rebinds_load_ckpt_in_upstream_modules(monkeypatch):
    def original(*a, **k):
        raise AssertionError("upstream load_ckpt called")

    for name in patch.MODULES:
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(load_ckpt=original))
    patch.apply()
    patch.apply()  # idempotent
    assert all(sys.modules[n].load_ckpt is patch.load_ckpt for n in patch.MODULES)
