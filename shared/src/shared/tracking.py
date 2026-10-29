"""W&B tracking: grouping, GPU cost, artifacts, timers.

Copied from `cartridges.cartridge.tracking` at commit a9a7588. Changed: the
project, run name, job type and entity are parameters of `start_run` (the
cartridge module fixed project "cartridges-selfstudy" and its own name and
job-type scheme); `log_artifact` is also a module function; the
cartridge-project budget roll-ups, Teacher attribution and W&B reports are not
copied. Git state in a Modal container comes from SHARED_GIT_SHA /
SHARED_GIT_DIRTY (`modal_env`).

    run = tracking.start_run("voice", tracking.group_name("smoke"), cfg.model_dump(),
                             name="personaplex/stock/eval-smoke", job_type="eval", gpu="H200")
    with run:
        with run.timer("step", tokens=n):
            ...
        run.log({"train/loss": loss})
        tracking.log_artifact(run, "out/cartridge", type="voice-cartridge", aliases=["best"])

Groups look like `exp-<YYYYMMDD>-<short-name>`. Every run records the config
hash, git SHA + dirty flag, Modal ids, GPU type and count, and its GPU spend
(rate x wall clock) in the summary as `cost/usd` and `cost/gpu_hours`,
refreshed on every `log()` so a crashed run still reports what it spent.

With WANDB_MODE=disabled (CPU tests) or offline everything still works:
logging is a no-op, cost is still computed, and `use_artifact` accepts local
paths.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

ENTITY = os.environ.get("WANDB_ENTITY", "conle-ai")

# Modal on-demand USD per GPU-hour (modal.com/pricing). "CPU" is a small CPU-only container.
GPU_RATES_USD_PER_HOUR = {
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
    "H200": 4.54,
    "B200": 6.25,
    "CPU": 0.05,
}

_GROUP_RE = re.compile(r"^exp-\d{8}-[a-z0-9][a-z0-9-]*$")
_REPO_ROOT = Path(__file__).resolve().parents[3]


# --- naming -------------------------------------------------------------------


def group_name(short_name: str, date: datetime.date | None = None) -> str:
    """`exp-<YYYYMMDD>-<short-name>`, date defaulting to today."""
    slug = re.sub(r"[^a-z0-9]+", "-", short_name.lower()).strip("-")
    if not slug:
        raise ValueError(f"empty experiment name: {short_name!r}")
    return f"exp-{(date or datetime.date.today()):%Y%m%d}-{slug}"


def config_hash(config: dict | None) -> str:
    """8 hex chars of sha256 over the canonical JSON of `config` (key order ignored)."""
    blob = json.dumps(config or {}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


# --- environment metadata -----------------------------------------------------


def git_info() -> dict:
    """{"git_sha", "git_dirty"} from the local checkout; inside a Modal container
    (no .git) from SHARED_GIT_SHA / SHARED_GIT_DIRTY, see `modal_env`."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
        return {"git_sha": sha, "git_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_sha": os.environ.get("SHARED_GIT_SHA", "unknown"),
            "git_dirty": os.environ.get("SHARED_GIT_DIRTY", "unknown") == "1",
        }


def modal_env() -> dict[str, str]:
    """Env vars carrying the launcher's git state into a Modal container. Attach with
    `secrets=[wandb_secret, modal.Secret.from_dict(tracking.modal_env())]`."""
    info = git_info()
    return {"SHARED_GIT_SHA": info["git_sha"], "SHARED_GIT_DIRTY": "1" if info["git_dirty"] else "0"}


def modal_info() -> dict:
    """Modal task/image/function-call ids when running in a Modal container, else {}."""
    if os.environ.get("MODAL_IS_REMOTE") != "1":
        return {}
    import modal

    return {
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
        "modal_region": os.environ.get("MODAL_REGION"),
        "modal_cloud": os.environ.get("MODAL_CLOUD_PROVIDER"),
        "modal_function_call_id": modal.current_function_call_id(),
    }


def normalize_gpu(name: str | None) -> tuple[str, int]:
    """Modal gpu= string ("A100-80GB", "h100:2") or a CUDA device name
    ("NVIDIA A100-SXM4-80GB") -> (rate-table key, count). None -> ("CPU", 1)."""
    if not name:
        return "CPU", 1
    base, _, count = name.partition(":")
    n = int(count) if count else 1
    upper = base.upper()
    if "A100" in upper:
        return ("A100-40GB" if "40GB" in upper else "A100-80GB"), n
    for key in ("H200", "H100", "B200", "L40S", "A10G", "L4", "T4"):
        if key in upper:
            return key, n
    if upper == "CPU":
        return "CPU", n
    raise ValueError(f"unknown GPU {name!r}; pass rate_usd_per_hour explicitly")


def detect_gpu() -> str | None:
    """CUDA device name of GPU 0, or None without torch/CUDA."""
    cuda = _cuda()
    return cuda.get_device_name(0) if cuda else None


def _cuda():
    """torch.cuda when a GPU is present, else None."""
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda if torch.cuda.is_available() else None


# --- the run ------------------------------------------------------------------


@dataclass
class Timer:
    seconds: float = 0.0


@dataclass
class TrackedRun:
    """A W&B run plus wall-clock cost accounting. Use as a context manager, or
    call `finish()` yourself."""

    run: Any  # wandb Run (a no-op one when WANDB_MODE=disabled)
    gpu: str
    n_gpus: int
    rate_usd_per_hour: float
    project: str
    entity: str = ENTITY
    clock: Callable[[], float] = time.monotonic
    start: float = field(default=0.0)

    def __post_init__(self):
        self.start = self.clock()

    @property
    def id(self) -> str:
        return self.run.id

    @property
    def name(self) -> str:
        return self.run.name

    @property
    def url(self) -> str | None:
        return getattr(self.run, "url", None)

    @property
    def summary(self):
        return self.run.summary

    @property
    def disabled(self) -> bool:
        return self.run.settings.mode == "disabled"

    def cost(self) -> dict:
        wall = self.clock() - self.start
        gpu_hours = wall / 3600 * self.n_gpus
        return {
            "cost/wall_clock_s": wall,
            "cost/gpu_hours": gpu_hours,
            "cost/usd": gpu_hours * self.rate_usd_per_hour,
        }

    def log(self, metrics: dict, step: int | None = None) -> None:
        self.run.log(metrics, step=step)
        self.run.summary.update(self.cost())

    @contextlib.contextmanager
    def timer(self, name: str, tokens: int | None = None, step: int | None = None) -> Iterator[Timer]:
        """Time a block (CUDA-synchronized on GPU) and log `time/<name>_s`, plus
        `throughput/<name>_tok_s` when `tokens` is given."""
        cuda = _cuda()
        result = Timer()
        if cuda:
            cuda.synchronize()
        t0 = time.perf_counter()
        yield result
        if cuda:
            cuda.synchronize()
        result.seconds = time.perf_counter() - t0
        metrics = {f"time/{name}_s": result.seconds}
        if tokens is not None and result.seconds > 0:
            metrics[f"throughput/{name}_tok_s"] = tokens / result.seconds
        self.log(metrics, step=step)

    def log_gpu_memory(self, step: int | None = None) -> dict:
        """Log torch's allocated/reserved/peak GPU memory (GB); {} without CUDA."""
        cuda = _cuda()
        if not cuda:
            return {}
        metrics = {
            "gpu/mem_alloc_gb": cuda.memory_allocated() / 1e9,
            "gpu/mem_reserved_gb": cuda.memory_reserved() / 1e9,
            "gpu/mem_peak_gb": cuda.max_memory_allocated() / 1e9,
        }
        self.log(metrics, step=step)
        return metrics

    def log_artifact(
        self,
        path: str | Path,
        type: str,
        name: str | None = None,
        metadata: dict | None = None,
        aliases: list[str] | None = None,
        table_columns: list[str] | None = None,
    ) -> str | None:
        """Upload a file or directory as a W&B artifact produced by this run.
        `name` defaults to `type` (so every version of a voice-cartridge is one
        artifact). With `table_columns`, a .jsonl file also goes in as a W&B
        table of those columns. Returns "<name>:latest", or None when tracking
        is disabled."""
        import wandb

        name = name or type
        if self.disabled:
            return None
        artifact = wandb.Artifact(name, type=type, metadata=metadata)
        path = Path(path)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        if table_columns:
            artifact.add(jsonl_table(path, table_columns), path.stem)
        self.run.log_artifact(artifact, aliases=aliases)
        return f"{name}:latest"

    def use_artifact(self, ref: str | Path, type: str | None = None, download: bool = True) -> Path | None:
        """Download an artifact as an input of this run (records lineage) and
        return its local directory. A local path is returned as-is.
        `download=False` only records the lineage and returns None."""
        if not download:
            if not self.disabled:
                self.run.use_artifact(qualify(str(ref), self.project, self.entity), type=type)
            return None
        if Path(ref).exists():
            return Path(ref)
        if self.disabled:
            raise FileNotFoundError(f"{ref} is not a local path and W&B is disabled")
        return Path(self.run.use_artifact(qualify(str(ref), self.project, self.entity), type=type).download())

    def finish(self, exit_code: int = 0) -> None:
        summary = self.cost()
        summary.update({k.replace("gpu/", "gpu/final_"): v for k, v in self.log_gpu_memory().items()})
        self.run.summary.update(summary)
        logger.info("run %s finished: $%.3f over %.0fs", self.name, summary["cost/usd"], summary["cost/wall_clock_s"])
        self.run.finish(exit_code=exit_code)

    def __enter__(self) -> TrackedRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(exit_code=1 if exc_type else 0)


def start_run(
    project: str,
    group: str,
    config: dict | None = None,
    *,
    name: str | None = None,
    job_type: str | None = None,
    gpu: str | None = None,
    rate_usd_per_hour: float | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    entity: str = ENTITY,
    clock: Callable[[], float] = time.monotonic,
) -> TrackedRun:
    """Start a W&B run in `project`.

    group:  from `group_name(...)`; shared by every run of one experiment.
    config: the run's full config (its hash goes into config["_meta"]).
    name:   the run name; the caller's project sets the scheme.
    gpu:    the Modal gpu= string ("H200", "H100:2"); detected from CUDA if
            omitted, CPU otherwise. Sets the cost rate unless `rate_usd_per_hour`.
    """
    import wandb

    if not _GROUP_RE.match(group):
        raise ValueError(f"group must look like exp-YYYYMMDD-name (use group_name()), got {group!r}")
    gpu_key, n_gpus = normalize_gpu(gpu or detect_gpu())
    rate = rate_usd_per_hour if rate_usd_per_hour is not None else GPU_RATES_USD_PER_HOUR[gpu_key]
    meta = {
        "config_hash": config_hash(config),
        "gpu": gpu_key,
        "n_gpus": n_gpus,
        "rate_usd_per_hour": rate,
        **git_info(),
        **modal_info(),
    }
    run = wandb.init(
        entity=entity,
        project=project,
        group=group,
        job_type=job_type,
        name=name,
        config={**(config or {}), "_meta": meta},
        tags=tags,
        notes=notes,
        settings=wandb.Settings(x_stats_sampling_interval=5.0),
    )
    if meta["git_dirty"] is True:
        logger.warning("W&B run %s started from a dirty git tree (%s)", run.name, meta["git_sha"][:8])
    return TrackedRun(run=run, gpu=gpu_key, n_gpus=n_gpus, rate_usd_per_hour=rate, project=project, entity=entity,
                      clock=clock)


def log_artifact(run: TrackedRun, path: str | Path, type: str, **kwargs) -> str | None:
    """`run.log_artifact(path, type, **kwargs)`."""
    return run.log_artifact(path, type, **kwargs)


def jsonl_table(path: str | Path, columns: list[str]):
    """A wandb.Table of `columns` from a JSONL file; lists and dicts become JSON
    strings, missing keys None."""
    import wandb

    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    cell = lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v  # noqa: E731
    return wandb.Table(columns=columns, data=[[cell(r.get(c)) for c in columns] for r in rows])


def fetch_artifact(ref: str | Path, project: str, type: str | None = None, entity: str = ENTITY) -> Path:
    """Download an artifact outside a run. A local path is returned as-is."""
    if Path(ref).exists():
        return Path(ref)
    import wandb

    return Path(wandb.Api().artifact(qualify(str(ref), project, entity), type=type).download())


def qualify(ref: str, project: str, entity: str = ENTITY) -> str:
    """"voice-cartridge" -> "<entity>/<project>/voice-cartridge:latest"."""
    if ":" not in ref:
        ref += ":latest"
    return ref if ref.count("/") == 2 else f"{entity}/{project}/{ref.rsplit('/', 1)[-1]}"
