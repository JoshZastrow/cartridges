"""Target Store: the Teacher's distillation targets as Parquet shards.

One row per labeled task (docs/plans/task-db.md §6): the student's input
(`prompt_ids`, the chat-templated task without the journal), the Teacher's
answer (`completion_ids`, ending with <|im_end|>) and its top-k next-token
logprobs at each answer token (`topk_ids`, `topk_logprobs`, [T, k]).
Full-vocabulary logits are not stored.

Shards live at <root>/<run_id>/part-NNNNN.parquet; `cartridge.labels` rows point
at (targets_uri, targets_row). The labels table is the source of truth: training
reads targets through `read_labeled(labels)`, which yields exactly the rows that
validated labels point at. A shard row without a labels row (e.g. a re-label the
store skipped on conflict) is never read. `read_examples` reads whole shards,
for inspection only.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cartridges.cartridge.tasks.types import Label, TargetRow

ROOT = "/vol/data/targets"  # on the cartridges-cartridge-cache Volume

SCHEMA = pa.schema([
    ("label_id", pa.string()),
    ("task_id", pa.string()),
    ("prompt_ids", pa.list_(pa.int32())),
    ("completion_ids", pa.list_(pa.int32())),
    ("topk_ids", pa.list_(pa.list_(pa.int32()))),
    ("topk_logprobs", pa.list_(pa.list_(pa.float32()))),
])


def shard_uri(run_id: str, part: int | str = 0, root: str | Path = ROOT) -> str:
    """<root>/<run_id>/part-<part>.parquet; an int part is zero-padded. label()
    names each batch's shard by its sample hash, so a resumed run with the same
    run_id never overwrites a shard that labels already point at."""
    name = f"{part:05d}" if isinstance(part, int) else part
    return f"{root}/{run_id}/part-{name}.parquet"


def write_shard(rows: Sequence[TargetRow], uri: str | Path) -> str:
    """Write rows to one Parquet shard (row i = targets_row i); returns the uri."""
    columns = {name: [getattr(row, name) for row in rows] for name in SCHEMA.names}
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(columns, schema=SCHEMA), path)
    return str(uri)


# Where apps mount the cartridges-cartridge-cache Volume: teacher.py at /cache,
# the distill app at /vol. Stored uris are made Volume-relative against these.
VOLUME_MOUNTS = ("/cache", "/vol")


def volume_relative(uri: str | Path) -> str | None:
    """"targets/<run>/part-x.parquet" for a uri under a known Volume mount (or
    already relative); None for any other absolute path (a local file)."""
    uri = str(uri)
    for mount in VOLUME_MOUNTS:
        if uri.startswith(f"{mount}/"):
            return uri[len(mount) + 1:]
    return None if uri.startswith("/") else uri


def resolve(uri: str | Path, root: str | Path | None = None) -> str:
    """Where to read a stored uri: under `root` (this process's mount of the
    Volume, e.g. "/vol", or a local copy) when it is Volume-relative or under a
    known mount; unchanged otherwise (or when root is None)."""
    rel = volume_relative(uri)
    return str(Path(root) / rel) if root is not None and rel is not None else str(uri)


def volume_writer(volume, mount: str = "/cache"):
    """A write_shard for a local orchestrator whose shards belong on a Modal
    Volume: writes the shard to a temp file, uploads it to the Volume, and
    returns the Volume-relative uri ("targets/<run_id>/part-<hash>.parquet"),
    which labels rows store; readers resolve it against their own mount
    (read_labeled(labels, root="/vol"))."""
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


def read_rows(uris: Iterable[str | Path], root: str | Path | None = None) -> Iterator[TargetRow]:
    for uri in uris:
        for record in pq.read_table(resolve(uri, root), schema=SCHEMA).to_pylist():
            yield TargetRow.model_validate(record)


def to_example(row: TargetRow):
    """A target row as `train_distill.Example` (id = label_id)."""
    from cartridges.cartridge.train_distill import Example

    return Example(
        id=row.label_id,
        prompt_ids=row.prompt_ids,
        completion_ids=row.completion_ids,
        topk_ids=row.topk_ids,
        topk_logprobs=row.topk_logprobs,
    )


def read_examples(uris: Iterable[str | Path], root: str | Path | None = None) -> Iterator:
    """Every row of the shards, labeled or not: for inspection, not training."""
    for row in read_rows(uris, root):
        yield to_example(row)


def read_labeled(labels: Iterable[Label], root: str | Path | None = None) -> Iterator:
    """The training read path: one `train_distill.Example` per validated label
    with a shard pointer, in the order of `labels`, each shard read once.
    `root` is where this process sees the Volume (e.g. "/vol" in the distill
    app): stored uris, Volume-relative or under any known mount ("/cache/…"
    from earlier runs), are read from there. Raises ValueError if a pointer
    lands on a row of another label."""
    labels = [lab for lab in labels if lab.status == "validated" and lab.targets_uri is not None]
    shards = {
        uri: pq.read_table(resolve(uri, root), schema=SCHEMA).to_pylist()
        for uri in dict.fromkeys(lab.targets_uri for lab in labels)
    }
    for lab in labels:
        row = TargetRow.model_validate(shards[lab.targets_uri][lab.targets_row])
        if row.label_id != str(lab.label_id):
            raise ValueError(f"{lab.targets_uri}[{lab.targets_row}] is label {row.label_id}, not {lab.label_id}")
        yield to_example(row)
