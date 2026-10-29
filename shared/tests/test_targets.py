"""shared.targets: Parquet top-k shards, one row per (id, frame, stream)."""

import pytest
from shared.targets import TargetRow, read_shards, resolve, shard_uri, write_shard


def row(frame: int, stream: str = "agent_text", id: str = "d0") -> TargetRow:
    return TargetRow(id=id, frame=frame, stream=stream, topk_ids=[1, 5], topk_logprobs=[-0.125, -2.5])


def test_round_trip_keeps_frame_and_stream(tmp_path):
    rows = [row(0), row(1), row(0, stream="agent", id="d1")]
    uri = write_shard(rows, shard_uri("run1", 3, tmp_path))
    assert uri.endswith("run1/part-00003.parquet")
    assert list(read_shards([uri])) == rows


def test_stream_defaults_to_text_and_extra_fields_rejected():
    assert TargetRow(id="d", frame=0, topk_ids=[], topk_logprobs=[]).stream == "text"
    with pytest.raises(ValueError):
        TargetRow(id="d", frame=0, topk_ids=[], topk_logprobs=[], task_id="t")


def test_resolve_volume_relative_uris(tmp_path):
    write_shard([row(0)], tmp_path / "targets/r/part-a.parquet")
    assert resolve("/vol/targets/r/part-a.parquet", root=tmp_path, mounts=["/vol"]) == str(tmp_path / "targets/r/part-a.parquet")
    assert resolve("/elsewhere/x.parquet", root=tmp_path, mounts=["/vol"]) == "/elsewhere/x.parquet"
    assert [r.id for r in read_shards(["targets/r/part-a.parquet"], root=tmp_path)] == ["d0"]
