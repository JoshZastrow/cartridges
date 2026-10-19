"""W&B tracking for the cartridge pipeline: naming, grouping, cost, artifacts, timers.

Every stage starts its run through `init_run` (never `wandb.init` directly), so
all runs follow the conventions in docs/plans/cartridge-pipeline.md:

    project  cartridges-selfstudy
    group    exp-<YYYYMMDD>-<short-name>          (one end-to-end experiment)
    job_type datagen | teacher | distill | grpo | eval | profile
    name     <intent>-<job_type>[-<role>]-<model-short>-<p>tok-<hash>
             (<intent> is the group's short name, the cartridge-train --name)

Each run records the config hash, git SHA + dirty flag, Modal app/function/task
ids, GPU type and count, and its GPU spend (rate x wall clock) in the summary as
`cost/usd`. The summary is refreshed on every `log()`, so a run that crashes
still reports what it spent. `group_costs` / `budget_status` roll spend up per
group and across the project against the ~$50 budget.

Metric names (so the group report lines up across stages):
    time/<name>_s, throughput/<name>_tok_s   from `TrackedRun.timer`
    gen/ttft_s, gen/decode_tok_s, ...        from `TrackedRun.log_generation`
    gpu/mem_{alloc,reserved,peak}_gb         from `TrackedRun.log_gpu_memory`
    cost/usd, cost/gpu_hours, cost/wall_clock_s
    eval/accuracy, eval/token_f1, eval/citation_validity, eval/ttft_s,
    eval/cache_mem_gb                        (eval runs; config["system"] is
                                              base | teacher | cartridge)

GPU utilization and memory over time come from W&B's own system metrics
(sampled every 5 s here), the same source `finetune.loop_eval_shard` uses.

Set WANDB_MODE=disabled (CPU tests) or offline and everything still works:
logging is a no-op, cost is still computed, and `use_artifact` accepts local
paths.

Usage:

    from cartridges.cartridge import tracking

    with tracking.init_run(
        "distill", "Qwen/Qwen3.5-4B", tracking.group_name("smoke"),
        config=cfg.model_dump(), p=512, gpu="A100-80GB",
    ) as run:
        with run.timer("step", tokens=n_tokens):
            loss = train_step(batch)
        run.log({"train/kl": loss})
        run.log_artifact("out/cartridge", type="cartridge")  # -> cartridge-<run_id>

Cost roll-up from the shell:
    uv run python -m cartridges.cartridge.tracking costs exp-20261013-smoke
    uv run python -m cartridges.cartridge.tracking budget
    uv run python -m cartridges.cartridge.tracking check 4.0   # exits non-zero if $4 more would pass $50

Before launching a GPU job, call `tracking.check_budget(estimate_usd)`; it raises
BudgetExceeded if tracked spend + the estimate would pass the budget. A run whose
finish sees tracked spend at 80% of the budget sends a `wandb.alert`.
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

PROJECT = "cartridges-selfstudy"
# Same team as finetune.WANDB_ENTITY (not imported: that module builds the OCR Modal image).
ENTITY = os.environ.get("WANDB_ENTITY", "conle-ai")
JOB_TYPES = ("datagen", "teacher", "distill", "grpo", "eval", "profile")
BUDGET_USD = 50.0
ALERT_FRACTION = 0.8  # wandb.alert once tracked spend reaches 80% of the budget

# Modal on-demand USD per GPU-hour (modal.com/pricing, matches the
# gpu-compute-planning estimator). "CPU" is a small CPU-only container.
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


def model_short(model_id: str) -> str:
    """"Qwen/Qwen3.5-4B" -> "qwen3.5-4b"."""
    return model_id.rsplit("/", 1)[-1].lower()


def config_hash(config: dict | None) -> str:
    """8 hex chars of sha256 over the canonical JSON of `config` (key order ignored)."""
    blob = json.dumps(config or {}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def run_name(job_type: str, model: str, p: int, config: dict | None, role: str | None = None,
             group: str | None = None) -> str:
    """`<intent>-<job_type>[-<role>]-<model-short>-<p>tok-<hash>`, where <intent> is
    the short name of `group` (exp-<date>-<intent>), so a run reads as what it was
    for. `p` is the cartridge length in tokens; 0 for stages without a cartridge
    (datagen, teacher, base eval)."""
    intent = group.split("-", 2)[2] if group else ""
    parts = [intent, job_type, role, model_short(model), f"{p}tok", config_hash(config)]
    return "-".join(x for x in parts if x)


# --- environment metadata -----------------------------------------------------


def git_info() -> dict:
    """{"git_sha", "git_dirty"} from the local checkout; inside a Modal container
    (no .git) from CARTRIDGES_GIT_SHA / CARTRIDGES_GIT_DIRTY, see `modal_env`."""
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
            "git_sha": os.environ.get("CARTRIDGES_GIT_SHA", "unknown"),
            "git_dirty": os.environ.get("CARTRIDGES_GIT_DIRTY", "unknown") == "1",
        }


def modal_env() -> dict[str, str]:
    """Env vars carrying the launcher's git state into a Modal container. Attach with
    `secrets=[wandb_secret, modal.Secret.from_dict(tracking.modal_env())]`."""
    info = git_info()
    return {"CARTRIDGES_GIT_SHA": info["git_sha"], "CARTRIDGES_GIT_DIRTY": "1" if info["git_dirty"] else "0"}


def modal_info() -> dict:
    """Modal app/function-call/task ids when running in a Modal container, else {}."""
    if os.environ.get("MODAL_IS_REMOTE") != "1":
        return {}
    import modal

    info = {
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
        "modal_region": os.environ.get("MODAL_REGION"),
        "modal_cloud": os.environ.get("MODAL_CLOUD_PROVIDER"),
        "modal_function_call_id": modal.current_function_call_id(),
    }
    with contextlib.suppress(Exception):  # private API; best effort
        info["modal_app_id"] = modal.App._get_container_app().app_id
    return info


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
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


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

    def log_generation(self, timing, prefix: str = "gen", step: int | None = None) -> None:
        """Log a `perf_metrics.GenerationTiming`: TTFT (= prefill), decode speed, total."""
        metrics = {
            f"{prefix}/ttft_s": timing.prefill_s,
            f"{prefix}/decode_s": timing.decode_s,
            f"{prefix}/generate_s": timing.generate_s,
            f"{prefix}/output_tokens": timing.n_output_tokens,
        }
        if timing.decode_tok_s is not None:
            metrics[f"{prefix}/decode_tok_s"] = timing.decode_tok_s
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
        `name` defaults to `<type>-<run_id>` (so cartridges are `cartridge-<run_id>`).
        With `table_columns`, a .jsonl file also goes in as a W&B table of those
        columns (`<stem>.table.json`), which the artifact's Files tab renders;
        a plain .jsonl is only downloadable.
        Returns "<name>:latest", or None when tracking is disabled."""
        import wandb

        name = name or f"{type}-{self.id}"
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
                self.run.use_artifact(_qualify(str(ref)), type=type)
            return None
        if Path(ref).exists():
            return Path(ref)
        if self.disabled:
            raise FileNotFoundError(f"{ref} is not a local path and W&B is disabled")
        return Path(self.run.use_artifact(_qualify(str(ref)), type=type).download())

    def finish(self, exit_code: int = 0) -> None:
        summary = self.cost()
        summary.update({k.replace("gpu/", "gpu/final_"): v for k, v in self.log_gpu_memory().items()})
        self.run.summary.update(summary)
        if not self.disabled and self.run.settings.mode != "offline":
            with contextlib.suppress(Exception):  # network roll-up is best effort
                group = group_costs(self.run.group)["total_usd"]
                project = budget_status()["spent_usd"]
                self.run.summary.update({"cost/group_usd": group, "cost/project_usd": project})
                if project > BUDGET_USD:
                    logger.warning("W&B project spend $%.2f exceeds the $%.0f budget", project, BUDGET_USD)
                if project >= ALERT_FRACTION * BUDGET_USD:
                    import wandb

                    self.run.alert(
                        title="cartridges-selfstudy budget",
                        text=f"Tracked GPU spend is ${project:.2f} of the ${BUDGET_USD:.0f} budget (run {self.name}).",
                        level=wandb.AlertLevel.WARN,
                        wait_duration=3600,  # at most one alert per hour
                    )
        logger.info("run %s finished: $%.3f over %.0fs", self.name, summary["cost/usd"], summary["cost/wall_clock_s"])
        self.run.finish(exit_code=exit_code)

    def __enter__(self) -> TrackedRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(exit_code=1 if exc_type else 0)


def init_run(
    job_type: str,
    model: str,
    group: str,
    config: dict | None = None,
    *,
    p: int = 0,
    gpu: str | None = None,
    rate_usd_per_hour: float | None = None,
    role: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> TrackedRun:
    """Start a W&B run that follows the team conventions.

    job_type: one of JOB_TYPES.
    model:    HF model id, e.g. "Qwen/Qwen3.5-4B".
    group:    from `group_name(...)`; shared by every stage of one experiment.
    config:   the stage's full config (hashed into the run name).
    p:        cartridge length in tokens (0 if none).
    gpu:      the Modal gpu= string ("A100-80GB", "H100:2"); detected from CUDA if
              omitted, CPU otherwise. Sets the cost rate unless `rate_usd_per_hour`.
    role:     what the run is within its stage, e.g. GRPO "gen" / "train", the eval
              system "base" / "teacher" / "cartridge".
    """
    import wandb

    if job_type not in JOB_TYPES:
        raise ValueError(f"job_type must be one of {JOB_TYPES}, got {job_type!r}")
    if not _GROUP_RE.match(group):
        raise ValueError(f"group must look like exp-YYYYMMDD-name (use group_name()), got {group!r}")
    gpu_key, n_gpus = normalize_gpu(gpu or detect_gpu())
    rate = rate_usd_per_hour if rate_usd_per_hour is not None else GPU_RATES_USD_PER_HOUR[gpu_key]
    meta = {
        "model": model,
        "p": p,
        "config_hash": config_hash(config),
        "gpu": gpu_key,
        "n_gpus": n_gpus,
        "rate_usd_per_hour": rate,
        **git_info(),
        **modal_info(),
    }
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=group,
        job_type=job_type,
        name=run_name(job_type, model, p, config, role, group),
        config={**(config or {}), "_meta": meta},
        tags=[job_type, model_short(model), *(tags or [])],
        notes=notes,
        settings=wandb.Settings(x_stats_sampling_interval=5.0),
    )
    if meta["git_dirty"] is True:
        logger.warning("W&B run %s started from a dirty git tree (%s)", run.name, meta["git_sha"][:8])
    return TrackedRun(run=run, gpu=gpu_key, n_gpus=n_gpus, rate_usd_per_hour=rate, clock=clock)


def jsonl_table(path: str | Path, columns: list[str]):
    """A wandb.Table of `columns` from a JSONL file; lists and dicts become JSON
    strings, missing keys None."""
    import wandb

    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    cell = lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v  # noqa: E731
    return wandb.Table(columns=columns, data=[[cell(r.get(c)) for c in columns] for r in rows])


def fetch_artifact(ref: str | Path, type: str | None = None) -> Path:
    """Download an artifact outside a run (e.g. `cartridge-ask --cartridge
    cartridge-<run_id>:latest`). A local path is returned as-is."""
    if Path(ref).exists():
        return Path(ref)
    import wandb

    return Path(wandb.Api().artifact(_qualify(str(ref)), type=type).download())


def _qualify(ref: str) -> str:
    """"cartridge-abc" -> "<entity>/cartridges-selfstudy/cartridge-abc:latest"."""
    if ":" not in ref:
        ref += ":latest"
    return ref if ref.count("/") == 2 else f"{ENTITY}/{PROJECT}/{ref.rsplit('/', 1)[-1]}"


# --- roll-ups and reports -----------------------------------------------------


def _runs(api, filters: dict | None = None) -> list:
    try:
        return list(api.runs(f"{ENTITY}/{PROJECT}", filters=filters or {}))
    except ValueError as e:  # the project is created by its first run
        if "Could not find project" in str(e):
            return []
        raise


def _run_cost(r) -> tuple[float, float]:
    """(usd, gpu_hours) of a W&B run. Runs that died before their first
    `log()` (e.g. in a Modal @enter) have no cost/* summary; for those, W&B's
    own `_runtime` times the rate recorded in config["_meta"]."""
    if "cost/usd" in r.summary:
        return float(r.summary["cost/usd"]), float(r.summary.get("cost/gpu_hours", 0.0))
    meta = r.config.get("_meta", {})
    gpu_hours = float(r.summary.get("_runtime", 0.0)) / 3600 * meta.get("n_gpus", 1)
    return gpu_hours * meta.get("rate_usd_per_hour", 0.0), gpu_hours


def group_costs(group: str, api=None) -> dict:
    """Spend of every run in `group`: {"total_usd", "by_job_type", "runs": [...]}."""
    import wandb

    rows = []
    for r in _runs(api or wandb.Api(), {"group": group}):
        usd, gpu_hours = _run_cost(r)
        rows.append({
            "name": r.name,
            "job_type": r.job_type,
            "state": r.state,
            "usd": usd,
            "gpu_hours": gpu_hours,
            "url": r.url,
        })
    by_job: dict[str, float] = {}
    for row in rows:
        by_job[row["job_type"]] = by_job.get(row["job_type"], 0.0) + row["usd"]
    direct = sum(by_job.values())
    teacher = teacher_attribution(api)["by_group"].get(group, 0.0)
    return {"group": group, "total_usd": direct, "teacher_service_usd": teacher,
            "total_with_teacher_usd": direct + teacher, "by_job_type": by_job, "runs": rows}


SERVICE_SUFFIX = "-teacher-service"


def _interval(r) -> tuple[float, float]:
    """[start, end] of a W&B run in epoch seconds: created_at plus its tracked wall clock."""
    start = datetime.datetime.fromisoformat(r.created_at.replace("Z", "+00:00")).timestamp()
    return start, start + float(r.summary.get("cost/wall_clock_s", r.summary.get("_runtime", 0.0)))


def teacher_attribution(api=None) -> dict:
    """Split the shared Teacher pool's spend (groups exp-<date>-teacher-service)
    over experiment groups by time overlap: each service run's cost is spread
    evenly over its lifetime, and every slice goes to the experiment groups with
    a run active at that moment (split equally if several are). Slices with no
    experiment active (startup before the first caller, scale-down idle) are
    `unattributed_usd`. Returns {"service_usd", "by_group", "unattributed_usd"}."""
    import wandb

    runs = _runs(api or wandb.Api())
    service = [r for r in runs if (r.group or "").endswith(SERVICE_SUFFIX)]
    windows: dict[str, list[tuple[float, float]]] = {}
    for r in runs:
        if r.group and not r.group.endswith(SERVICE_SUFFIX):
            windows.setdefault(r.group, []).append(_interval(r))
    by_group: dict[str, float] = {}
    unattributed = 0.0
    for t in service:
        (t0, t1), usd = _interval(t), _run_cost(t)[0]
        if t1 <= t0:
            unattributed += usd
            continue
        cuts = sorted({t0, t1, *(x for ivs in windows.values() for iv in ivs for x in iv if t0 < x < t1)})
        for a, b in zip(cuts, cuts[1:]):
            share = usd * (b - a) / (t1 - t0)
            active = [g for g, ivs in windows.items() if any(s <= a and b <= e for s, e in ivs)]
            if not active:
                unattributed += share
            for g in active:
                by_group[g] = by_group.get(g, 0.0) + share / len(active)
    return {"service_usd": sum(_run_cost(t)[0] for t in service), "by_group": by_group, "unattributed_usd": unattributed}


def budget_status(api=None) -> dict:
    """Project-wide spend against BUDGET_USD, split by group."""
    import wandb

    by_group: dict[str, float] = {}
    for r in _runs(api or wandb.Api()):
        by_group[r.group or "-"] = by_group.get(r.group or "-", 0.0) + _run_cost(r)[0]
    spent = sum(by_group.values())
    return {"spent_usd": spent, "budget_usd": BUDGET_USD, "remaining_usd": BUDGET_USD - spent, "by_group": by_group}


class BudgetExceeded(RuntimeError):
    pass


def check_budget(estimate_usd: float, api=None) -> dict:
    """Pre-launch guard: raise BudgetExceeded if tracked spend + `estimate_usd`
    would pass BUDGET_USD; else return `budget_status()`. Tracked spend is a
    lower bound (GPU only, no CPU/memory), so leave some margin."""
    status = budget_status(api)
    if status["spent_usd"] + estimate_usd > BUDGET_USD:
        raise BudgetExceeded(
            f"${status['spent_usd']:.2f} spent + ${estimate_usd:.2f} estimate > ${BUDGET_USD:.0f} budget"
        )
    logger.info("budget ok: $%.2f spent + $%.2f estimate of $%.0f", status["spent_usd"], estimate_usd, BUDGET_USD)
    return status


EVAL_METRICS = ("eval/accuracy", "eval/token_f1", "eval/citation_validity", "eval/ttft_s", "eval/cache_mem_gb")


def group_summary_markdown(group: str, api=None) -> str:
    """Markdown summary of a group: every stage's runs with cost, and the eval
    comparison across systems (base / teacher / cartridge)."""
    import wandb

    runs = _runs(api or wandb.Api(), {"group": group})
    total = sum(_run_cost(r)[0] for r in runs)
    teacher = teacher_attribution(api)["by_group"].get(group, 0.0)
    lines = [f"## {group}", "", f"GPU spend: ${total:.2f} direct + ${teacher:.2f} shared Teacher service "
             f"(time-window share) = ${total + teacher:.2f} of ${BUDGET_USD:.0f} budget", ""]
    lines += ["| job_type | run | state | GPU | cost (USD) |", "|---|---|---|---|---|"]
    order = {j: i for i, j in enumerate(JOB_TYPES)}
    for r in sorted(runs, key=lambda r: (order.get(r.job_type, 99), r.name)):
        gpu = r.config.get("_meta", {}).get("gpu", "?")
        lines.append(f"| {r.job_type} | [{r.name}]({r.url}) | {r.state} | {gpu} | {_run_cost(r)[0]:.3f} |")
    evals = [r for r in runs if r.job_type == "eval" and "system" in r.config]  # not cartridge-ask runs
    if evals:
        lines += ["", "| system | " + " | ".join(m.removeprefix("eval/") for m in EVAL_METRICS) + " |"]
        lines.append("|---" * (len(EVAL_METRICS) + 1) + "|")
        for r in evals:
            cells = [r.summary.get(m) for m in EVAL_METRICS]
            lines.append(
                f"| {r.config['system']} | "
                + " | ".join("-" if c is None else f"{c:.4g}" for c in cells)
                + " |"
            )
    return "\n".join(lines) + "\n"


def create_group_report(group: str) -> str:
    """Save a W&B report for `group` (stage panels, eval comparison, cost) and
    return its URL. Needs `wandb-workspaces`:
    uv run --with wandb-workspaces python -m cartridges.cartridge.tracking report <group>"""
    import wandb_workspaces.reports.v2 as wr

    def runset(name: str, extra: str = "") -> wr.Runset:
        return wr.Runset(entity=ENTITY, project=PROJECT, name=name, filters=f"Metric('Group') == '{group}'{extra}")

    report = wr.Report(
        project=PROJECT,
        entity=ENTITY,
        title=f"Cartridge experiment {group}",
        description="All pipeline stages for one experiment: datagen, teacher, distill, GRPO, eval.",
        blocks=[
            wr.H1(text="Cost"),
            wr.PanelGrid(
                runsets=[runset("all stages")],
                panels=[
                    wr.BarPlot(title="GPU spend per run (USD)", metrics=["cost/usd"]),
                    wr.ScalarChart(title="Group spend (USD)", metric="cost/usd", groupby_aggfunc="sum"),
                ],
            ),
            wr.H1(text="Stage 1: self-study distillation"),
            wr.PanelGrid(
                runsets=[runset("distill", " and Metric('JobType') == 'distill'")],
                panels=[wr.LinePlot(y=["train/kl"]), wr.LinePlot(y=["eval/accuracy"]), wr.LinePlot(y=["throughput/step_tok_s"]), wr.LinePlot(y=["gpu/mem_peak_gb"])],
            ),
            wr.H1(text="Stage 2: async GRPO"),
            wr.PanelGrid(
                runsets=[runset("grpo", " and Metric('JobType') == 'grpo'")],
                panels=[wr.LinePlot(y=["grpo/reward_mean"]), wr.LinePlot(y=["eval/accuracy"]), wr.LinePlot(y=["grpo/kl"]), wr.LinePlot(y=["grpo/clip_frac"]), wr.LinePlot(y=["grpo/staleness_mean", "grpo/staleness_max"]), wr.LinePlot(y=["throughput/step_tok_s"])],
            ),
            wr.H1(text="Eval: base vs teacher (ICL) vs cartridge"),
            wr.PanelGrid(
                runsets=[runset("eval", " and Metric('JobType') == 'eval'")],
                panels=[wr.BarPlot(title=m, metrics=[m]) for m in EVAL_METRICS] + [wr.RunComparer(diff_only="split")],
            ),
        ],
    )
    report.save()
    return report.url


# 27B phase: one group per stage (agreed with the lead, 2026-10-16).
PHASE_STAGES = ("label-27b", "train-27b-small", "sweep-27b", "train-27b", "eval-27b")


def phase_groups(date: str = "20261016", stages: tuple[str, ...] = PHASE_STAGES) -> dict[str, str]:
    """{stage: "exp-<date>-<stage>"}."""
    return {s: f"exp-{date}-{s}" for s in stages}


def stage_cost_markdown(groups: dict[str, str], api=None) -> str:
    """Per-stage spend: direct GPU/CPU runs + time-window share of the Teacher service."""
    lines = ["| stage | group | runs | direct (USD) | Teacher service (USD) | total (USD) |", "|---|---|---|---|---|---|"]
    total = 0.0
    for stage, group in groups.items():
        c = group_costs(group, api)
        total += c["total_with_teacher_usd"]
        lines.append(f"| {stage} | `{group}` | {len(c['runs'])} | {c['total_usd']:.2f} | "
                     f"{c['teacher_service_usd']:.2f} | {c['total_with_teacher_usd']:.2f} |")
    lines.append(f"| **all** | | | | | **{total:.2f}** |")
    return "\n".join(lines)


def build_phase_report(groups: dict[str, str], api=None):
    """W&B report for the 27B phase (not saved): sweep curves, scale-up with the
    early-stop point, eval CGC headline and tables, per-stage cost (a snapshot at
    build time -- rebuild to refresh)."""
    import wandb_workspaces.reports.v2 as wr

    def runset(stage: str) -> wr.Runset:
        return wr.Runset(entity=ENTITY, project=PROJECT, name=stage, filters=f"Metric('Group') == '{groups[stage]}'")

    def stack(panels: list) -> list:
        """Full page width, one panel per row (default is a third of the 24-column grid)."""
        y = 0
        for panel in panels:
            h = 10 if isinstance(panel, (wr.WeavePanelSummaryTable, wr.RunComparer)) else 8
            panel.layout = wr.Layout(x=0, y=y, w=24, h=h)
            y += h
        return panels

    def curves(legend: str | None = None) -> list:
        return [wr.LinePlot(title=k, y=[k], legend_template=legend) for k in ("train/kl", "val/kl", "train/grad_norm")]

    return wr.Report(
        project=PROJECT,
        entity=ENTITY,
        title="Cartridge 27B: labeling, LR sweep, scale-up, eval",
        description="Groups: " + ", ".join(groups.values()),
        blocks=[
            wr.H1(text="LR sweep"),
            wr.P(text="train KL, val KL and grad norm vs step, one line per sweep run (legend: learning rate)."),
            wr.PanelGrid(runsets=[runset("sweep-27b")], panels=stack(curves("lr=${config:lr}"))),
            wr.H1(text="Small training run"),
            wr.PanelGrid(runsets=[runset("train-27b-small")], panels=stack(curves())),
            wr.H1(text="Scale-up with early stopping"),
            wr.P(text="x = val/step. The early-stop point is the marker early_stop/val_kl, the val KL at the restored best step (early_stop/step; reason in early_stop/reason)."),
            wr.PanelGrid(
                runsets=[runset("train-27b")],
                panels=stack([
                    wr.LinePlot(title="val KL + early stop", x="val/step", y=["val/kl", "val/best_kl", "early_stop/val_kl"],
                                line_marks={"early_stop/val_kl": "points"}),
                    wr.ScalarChart(title="early stop step", metric="early_stop/step"),
                    wr.LinePlot(title="train/kl", y=["train/kl"]),
                    wr.LinePlot(title="train/grad_norm", y=["train/grad_norm"]),
                    wr.RunComparer(diff_only="split"),
                ]),
            ),
            wr.H1(text="Eval: CGC (context gap closed)"),
            wr.P(text="CGC = (NLL_base - NLL_cart) / (NLL_base - NLL_teacher) on the teacher's answers to held-out "
                      "tasks: 1.0 = the cartridge explains them as well as the full journal, 0 = no better than base. "
                      "One eval run per cartridge. Rows with small = true have fewer than 10 tasks."),
            wr.PanelGrid(
                runsets=[runset("eval-27b")],
                panels=stack([
                    wr.BarPlot(title="CGC with 95% CI", metrics=["eval/cgc", "eval/cgc_ci_low", "eval/cgc_ci_high"]),
                    wr.BarPlot(title="Perplexity: base / cartridge / teacher",
                               metrics=["eval/ppl_base", "eval/ppl_cart", "eval/ppl_teacher"]),
                    wr.WeavePanelSummaryTable(table_name="eval/by_source"),
                    wr.WeavePanelSummaryTable(table_name="eval/by_kind"),
                ]),
            ),
            wr.H1(text="Cost per stage"),
            wr.MarkdownBlock(text=stage_cost_markdown(groups, api)),
            wr.PanelGrid(
                runsets=[runset(s) for s in groups],
                panels=stack([wr.BarPlot(title="Direct cost per run (USD)", metrics=["cost/usd"])]),
            ),
        ],
    )


def create_phase_report(date: str = "20261016") -> str:
    """Save the 27B phase report and return its URL:
    uv run python -m cartridges.cartridge.tracking phase-report"""
    report = build_phase_report(phase_groups(date))
    report.save()
    return report.url


def main() -> None:
    import fire

    fire.Fire({
        "costs": lambda group: json.dumps(group_costs(group), indent=2),
        "budget": lambda: json.dumps(budget_status(), indent=2),
        "teacher": lambda: json.dumps(teacher_attribution(), indent=2),
        "check": lambda estimate_usd: json.dumps(check_budget(float(estimate_usd)), indent=2),
        "summary": group_summary_markdown,
        "report": create_group_report,
        "phase-report": create_phase_report,
        "stage-costs": lambda date="20261016": stage_cost_markdown(phase_groups(str(date))),
    })


if __name__ == "__main__":
    main()
