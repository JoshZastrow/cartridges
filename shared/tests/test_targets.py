"""shared.targets: Parquet top-k shards with a stream column."""

import pytest
from shared.targets import TargetRow, read_shards, resolve, shard_uri, write_shard


def row(i: int, stream: str = "agent_text") -> TargetRow:
    return TargetRow(label_id=f"l{i}", task_id=f"d{i}", stream=stream, prompt_ids=[], completion_ids=[1, 2],
                     topk_ids=[[1, 5], [2, 7]], topk_logprobs=[[-0.125, -2.5], [-0.25, -1.75]])


def test_round_trip_keeps_stream(tmp_path):
    rows = [row(0), row(1, stream="agent")]
    uri = write_shard(rows, shard_uri("run1", 3, tmp_path))
    assert uri.endswith("run1/part-00003.parquet")
    assert list(read_shards([uri])) == rows


def test_stream_defaults_to_text_and_extra_fields_rejected():
    assert TargetRow(label_id="l", task_id="t", prompt_ids=[1], completion_ids=[], topk_ids=[], topk_logprobs=[]).stream == "text"
    with pytest.raises(ValueError):
        TargetRow(label_id="l", task_id="t", prompt_ids=[], completion_ids=[], topk_ids=[], topk_logprobs=[], x=1)


def test_resolve_volume_relative_uris(tmp_path):
    write_shard([row(0)], tmp_path / "targets/r/part-a.parquet")
    assert resolve("/vol/targets/r/part-a.parquet", root=tmp_path, mounts=["/vol"]) == str(tmp_path / "targets/r/part-a.parquet")
    assert resolve("/elsewhere/x.parquet", root=tmp_path, mounts=["/vol"]) == "/elsewhere/x.parquet"
    assert [r.label_id for r in read_shards(["targets/r/part-a.parquet"], root=tmp_path)] == ["l0"]
