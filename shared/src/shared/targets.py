"""Target shards: a teacher's top-k distillation targets as Parquet.

Copied from `cartridges.cartridge.tasks.targets` (and `TargetRow` from
`cartridges.cartridge.tasks.types`) at commit a9a7588. Changed: `TargetRow`
has a `stream` column, so one shard can hold targets for several output
streams of a multi-stream model (the voice project's `agent_text` or `agent`
FSQ stream; "text" for a plain LM); the shard root and the Volume mounts are
parameters instead of the cartridge Volume's paths; the cartridge-only
label/Example readers are not copied.

One row per (sequence, stream):

    label_id, task_id   str        ids of the target and of the input it was made for
    stream              str        output stream the targets predict
    prompt_ids          [P] int32  the student's input tokens ([] when the input is not tokens)
    completion_ids      [T] int32  the target tokens on `stream`
    topk_ids            [T, k] int32    teacher's top-k ids of the distribution predicting completion_ids[t]
    topk_logprobs       [T, k] float32  their logprobs

Position t is frame t for a frame-synchronous stream. Full-vocabulary logits
are not stored. Shards live at <root>/<run_id>/part-NNNNN.parquet.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict


class TargetRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label_id: str
    task_id: str
    stream: str = "text"
    prompt_ids: list[int]
    completion_ids: list[int]
    topk_ids: list[list[int]]
    topk_logprobs: list[list[float]]


SCHEMA = pa.schema([
    ("label_id", pa.string()),
    ("task_id", pa.string()),
    ("stream", pa.string()),
    ("prompt_ids", pa.list_(pa.int32())),
    ("completion_ids", pa.list_(pa.int32())),
    ("topk_ids", pa.list_(pa.list_(pa.int32()))),
    ("topk_logprobs", pa.list_(pa.list_(pa.float32()))),
])


def shard_uri(run_id: str, part: int | str, root: str | Path) -> str:
    """<root>/<run_id>/part-<part>.parquet; an int part is zero-padded. Name a
    batch's shard by a hash of its inputs so a resumed run never overwrites a
    shard that something already points at."""
    name = f"{part:05d}" if isinstance(part, int) else part
    return f"{root}/{run_id}/part-{name}.parquet"


def write_shard(rows: Sequence[TargetRow], uri: str | Path) -> str:
    """Write rows to one Parquet shard (row i = targets_row i); returns the uri."""
    columns = {name: [getattr(row, name) for row in rows] for name in SCHEMA.names}
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(columns, schema=SCHEMA), path)
    return str(uri)


def volume_relative(uri: str | Path, mounts: Sequence[str]) -> str | None:
    """"targets/<run>/part-x.parquet" for a uri under one of `mounts` (or
    already relative); None for any other absolute path (a local file)."""
    uri = str(uri)
    for mount in mounts:
        if uri.startswith(f"{mount}/"):
            return uri[len(mount) + 1:]
    return None if uri.startswith("/") else uri


def resolve(uri: str | Path, root: str | Path | None = None, mounts: Sequence[str] = ()) -> str:
    """Where to read a stored uri: under `root` (this process's mount of the
    Volume, or a local copy) when it is Volume-relative or under one of
    `mounts`; unchanged otherwise (or when root is None)."""
    rel = volume_relative(uri, mounts)
    return str(Path(root) / rel) if root is not None and rel is not None else str(uri)


def volume_writer(volume, mount: str):
    """A write_shard for a local orchestrator whose shards belong on a Modal
    Volume mounted at `mount` in the readers: writes the shard to a temp file,
    uploads it, and returns the Volume-relative uri, which readers resolve
    against their own mount (read_shards(uris, root=mount))."""
    import tempfile

    def write(rows: Sequence[TargetRow], uri: str | Path) -> str:
        uri = str(uri)
        if not uri.startswith(f"{mount}/"):
            raise ValueError(f"{uri} is not under the Volume mount {mount}")
        with tempfile.TemporaryDirectory() as tmp:
            local = write_shard(rows, Path(tmp) / "part.parquet")
            with volume.batch_upload(force=True) as batch:
                batch.put_file(local, uri[len(mount):])
        return uri[len(mount) + 1:]

    return write


def read_shards(
    uris: Iterable[str | Path], root: str | Path | None = None, mounts: Sequence[str] = ()
) -> Iterator[TargetRow]:
    """Every row of the shards, in order."""
    for uri in uris:
        for record in pq.read_table(resolve(uri, root, mounts), schema=SCHEMA).to_pylist():
            yield TargetRow.model_validate(record)
