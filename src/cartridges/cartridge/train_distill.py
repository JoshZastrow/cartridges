"""Stage 1: self-study context distillation into a cartridge.

The student is the frozen base model reading the cartridge (cache.Cartridge)
instead of the timeline. For every self-study conversation (data.py), the
teacher scored the assistant turn with the full timeline in context and kept
its top-k next-token logprobs. The loss is the paper's cross-entropy against
that top-k distribution on the completion tokens,

    loss = -mean_t sum_k exp(teacher_lp[t, k]) * student_lp[t, topk_ids[t, k]]

which is KL(teacher_topk || student) up to the teacher's (constant) entropy.
Only the cartridge's tensors receive gradients.

Hyperparameters follow HazyResearch/cartridges: Adam, lr 2e-2, top-k 20,
global batch 32 conversations, one epoch, p = 2048. Sequences are not packed:
the linear-attention state would carry from one conversation into the next.

Run on Modal with `uv run cartridges cartridge_train` (cli.py), or directly:
uv run modal run src/cartridges/cartridge/train_distill.py
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import modal
from pydantic import BaseModel, ConfigDict, model_validator

from cartridges.cartridge import tracking

logger = logging.getLogger(__name__)

MINUTES = 60
VOLUME_PATH = "/vol"
JOURNALS_DIR = "data/journals/output/2026_10_13"
JOURNALS_PATH = "/journals"
GPU = "A100-80GB"

image = (
    # CUDA devel base: causal-conv1d ships source only and compiles against torch with nvcc;
    # the toolkit must match torch 2.14.1's CUDA (13.0).
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.17.0",  # same as teacher.py
        "accelerate",
        "flash-linear-attention==0.5.2",
        "safetensors",
        "huggingface_hub",
        "hf_transfer",
        "pydantic>=2",
        "wandb",
        "fire",
        "rapidfuzz",
        "pyarrow",  # tasks/targets.py shards
        "psycopg[binary]",  # tasks/store.py labels
        "ninja",
        "packaging",
    )
    # Fused DeltaNet conv kernels (prefill + per-token decode update); without it HF
    # falls back to a PyTorch reference conv. sm80 = A100, sm90 = H100.
    .env({"TORCH_CUDA_ARCH_LIST": "8.0;9.0", "MAX_JOBS": "8", "CAUSAL_CONV1D_FORCE_BUILD": "TRUE",
          "CC": "gcc", "CXX": "g++"})
    .run_commands("python -m pip install --no-build-isolation causal-conv1d==1.7.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": f"{VOLUME_PATH}/hf"})
    .add_local_dir(JOURNALS_DIR, JOURNALS_PATH)  # transcripts + teacher_context.md: cartridge init, citation checks
    .add_local_python_source("cartridges")
)
app = modal.App("cartridges-distill", image=image)
cpu_app = modal.App("cartridges-distill-cpu", image=image)  # dry runs: no GPU
volume = modal.Volume.from_name("cartridges-cartridge-cache", create_if_missing=True)
secrets = [
    modal.Secret.from_name("wandb-secret"),
    modal.Secret.from_name("HF_TOKEN"),
    modal.Secret.from_name("cartridges-db"),  # DATABASE_URL for the Task Store labels
    modal.Secret.from_dict(tracking.modal_env()),
]


class DistillConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = "Qwen/Qwen3.5-4B"
    p: int = 2048
    lr: float = 2e-2
    batch_size: int = 32  # conversations per optimizer step (gradient accumulation)
    epochs: int = 1
    max_steps: int = 0  # 0: all of `epochs`
    # Conversation tokens after the cartridge: prompt + up to 3072 label tokens
    # (teacher max_answer_tokens). Longer ones lose the end of their completion
    # (counted in data/truncated_examples).
    max_seq_len: int = 3584
    # Targets: validated labels of these Teacher label runs (Task Store + Parquet
    # shards, tasks/targets.read_labeled). Empty: the legacy self-study JSONL at train_path.
    label_runs: list[str] = []
    # Batches drawn by task origin (with replacement; taxonomy.LABEL_ORIGIN_MIX
    # for real runs), `samples` conversations in all = samples / batch_size steps.
    # Empty: `epochs` uniform passes.
    origin_mix: dict[str, float] = {}
    samples: int = 0
    # Validation KL (the main learning curve): teacher labels of split='eval'
    # tasks, a separate label set never trained on; every val_every steps.
    val_label_runs: list[str] = []
    val_every: int = 10
    val_max: int = 0  # >0: a fixed, origin-stratified subset of this many val examples
    # Exact validation labels (label_ids, from an earlier run's data/val_label_ids):
    # keeps the val set identical across runs while the label pool changes.
    val_label_ids: list[str] = []
    # Early stopping: stop after `patience` val evals without a new best val KL
    # (0: never), and end with the best checkpoint (<out_dir>/distill-best).
    patience: int = 0
    # Resume stage 1 from a saved cartridge (Adam moments start fresh: use warmup).
    # skip_samples: draws of the origin-weighted order already trained on (the
    # resumed run continues the same order; steps are logged from skip/batch).
    # best_val_kl: the checkpoint's val KL, so early stopping only saves a better one.
    resume_from: str = ""
    skip_samples: int = 0
    best_val_kl: float | None = None
    # Only labels made with these teacher instructions (label.instructions_hash());
    # empty: the current hash. Earlier pilots' labels carry other hashes.
    instructions_hash: str = ""
    warmup_steps: int = 0

    @model_validator(mode="after")
    def _check_mix(self) -> "DistillConfig":
        if self.origin_mix and (self.samples <= 0 or not self.label_runs):
            raise ValueError("origin_mix needs label_runs and samples > 0")
        return self
    train_path: str = f"{VOLUME_PATH}/data/selfstudy/train.jsonl"
    init_text_path: str = f"{JOURNALS_PATH}/teacher_context.md"
    out_dir: str = f"{VOLUME_PATH}/cartridges/dev"
    save_every: int = 0  # optimizer steps; 0: only at the end
    # Per-tensor lr = lr x initial RMS (see train()). 4B sweep on the smoke data,
    # per-batch KL on unseen batches (init 1.10): absolute 2e-2 jumps to 6.3 and
    # ends 1.35; absolute 2e-3 ends 0.87; relative 2e-2 is lowest at every step, ends 0.80.
    relative_lr: bool = True
    group: str = ""
    seed: int = 0


# ---------------------------------------------------------------- data


class Example(BaseModel):
    """One scored conversation. `prompt_ids` is the chat-templated
    conversation up to the assistant turn, without the timeline; the teacher's
    distribution at completion token t is (topk_ids[t], topk_logprobs[t])."""

    model_config = ConfigDict(extra="ignore")

    id: str
    prompt_ids: list[int]
    completion_ids: list[int]
    topk_ids: list[list[int]]
    topk_logprobs: list[list[float]]


def load_examples(path: str | Path) -> list[Example]:
    with open(path) as f:
        return [Example.model_validate_json(line) for line in f if line.strip()]


def to_tensors(example: Example, max_seq_len: int):
    """(input_ids [1, L], topk_ids [T, k], topk_logprobs [T, k]); the
    completion is the last T tokens of input_ids. A conversation longer than
    `max_seq_len` loses the end of its completion."""
    import torch

    T = min(len(example.completion_ids), max_seq_len - len(example.prompt_ids))
    if T <= 0:
        raise ValueError(f"{example.id}: prompt alone is {len(example.prompt_ids)} tokens")
    ids = example.prompt_ids + example.completion_ids[:T]
    return (
        torch.tensor([ids]),
        torch.tensor(example.topk_ids[:T]),
        torch.tensor(example.topk_logprobs[:T], dtype=torch.float32),
    )


# ---------------------------------------------------------------- loss


def distill_loss(logits, topk_ids, topk_logprobs) -> tuple:
    """Cross-entropy of the student against the teacher's top-k, mean over
    completion tokens. `logits` [T, V] are the student's predictions for the T
    completion tokens. Returns (loss, kl): kl = KL(teacher_topk || student)
    over the top-k entries, for logging."""
    import torch

    student = torch.log_softmax(logits.float(), dim=-1).gather(-1, topk_ids)
    teacher = topk_logprobs.exp()
    loss = -(teacher * student).sum(-1).mean()
    kl = (teacher * (topk_logprobs - student)).sum(-1).mean()
    return loss, kl.detach()


def student_logits(model, cartridge, input_ids, T: int):
    """The student's logits [T, V] for the last T tokens of `input_ids`, read
    after the cartridge (positions p..p+L-1)."""
    import torch

    L = input_ids.shape[1]
    position_ids = torch.arange(cartridge.p, cartridge.p + L, device=input_ids.device)[None]
    out = model(
        input_ids=input_ids,
        past_key_values=cartridge.as_past_key_values(1),
        position_ids=position_ids,
        use_cache=False,
        logits_to_keep=T + 1,  # position i predicts token i+1
    )
    return out.logits[0, :-1]


def student_loss(model, cartridge, input_ids, topk_ids, topk_logprobs, chunk: int = 1024) -> tuple:
    """distill_loss(student_logits(...)) without materializing [T, V] logits:
    lm_head + log_softmax run over `chunk` completion positions at a time,
    each chunk checkpointed (recomputed in backward). At T = 3072 and a 248k
    vocab, full fp32 logits plus log_softmax and their grad are ~10 GB; a
    1024-token chunk is ~3 GB."""
    import torch
    from torch.utils.checkpoint import checkpoint

    T, L = topk_ids.shape[0], input_ids.shape[1]
    position_ids = torch.arange(cartridge.p, cartridge.p + L, device=input_ids.device)[None]
    hidden = model.model(
        input_ids=input_ids, past_key_values=cartridge.as_past_key_values(1), position_ids=position_ids,
        use_cache=False,
    ).last_hidden_state[0, L - T - 1 : L - 1]  # position i predicts token i+1

    def sums(h, ids, lp):
        student = torch.log_softmax(model.lm_head(h).float(), dim=-1).gather(-1, ids)
        teacher = lp.exp()
        return -(teacher * student).sum(), (teacher * (lp - student)).sum()

    loss = kl = 0.0
    for a in range(0, T, chunk):
        args = (hidden[a : a + chunk], topk_ids[a : a + chunk], topk_logprobs[a : a + chunk])
        l, k = checkpoint(sums, *args, use_reentrant=False) if torch.is_grad_enabled() else sums(*args)
        loss, kl = loss + l, kl + k.detach()
    return loss / T, kl / T


# ---------------------------------------------------------------- training


def prepare_frozen(model, gradient_checkpointing: bool = True) -> None:
    """Freeze the base model and turn on non-reentrant gradient checkpointing.
    The checkpointed layers keep the cartridge: HF drops past_key_values under
    checkpointing unless a layer says the cache is read-only, which the
    cartridge's cache is (its updates return prefix+new without storing)."""
    model.requires_grad_(False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for layer in _decoder_layers(model):
            layer._can_checkpoint_with_cache = True
    model.train()  # HF only checkpoints in train mode; qwen3_5 has no dropout


def _decoder_layers(model):
    for module in model.modules():
        if type(module).__name__.endswith("DecoderLayer"):
            yield module


def train(
    model,
    cartridge,
    examples: list[Example],
    *,
    lr: float = 2e-2,
    batch_size: int = 32,
    epochs: int = 1,
    max_steps: int = 0,
    max_seq_len: int = 3584,
    seed: int = 0,
    relative_lr: bool = False,
    order: list[int] | None = None,
    warmup_steps: int = 0,
    log: Callable[[dict, int], None] = lambda metrics, step: None,
    on_step: Callable[[int], bool | None] = lambda step: None,
) -> dict:
    """Optimize the cartridge's trainable tensors on `examples`. Calls
    `log(metrics, step)` and `on_step(step)` after every optimizer step; a
    truthy on_step return stops training (early stopping). Raises
    FloatingPointError on a non-finite loss or gradient, before the update.
    Returns the last step's metrics.

    order: the example indices to visit, in order (e.g. weighted_order());
    default `epochs` shuffled passes. warmup_steps: linear lr warmup from
    lr/warmup_steps, then constant.

    relative_lr: each tensor's lr is `lr` x its initial RMS. Adam moves every
    entry by about lr per step, which suits the paper's KV prefix (RMS ~1) but
    is large for qwen3_5's DeltaNet recurrent states (RMS ~0.05 on 4B)."""
    import torch

    params = [x for x in cartridge.parameters() if x.requires_grad]
    if relative_lr:
        groups = [{"params": [x], "lr": lr * x.detach().pow(2).mean().sqrt().clamp(min=1e-3).item()} for x in params]
        optimizer = torch.optim.Adam(groups, lr=lr)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(warmup_steps, 1)))
    device = params[0].device
    order = order if order is not None else _epoch_order(len(examples), epochs, seed)
    steps = math.ceil(len(order) / batch_size)
    if max_steps:
        steps = min(steps, max_steps)

    metrics: dict = {}
    for step in range(1, steps + 1):
        batch = [examples[i] for i in order[(step - 1) * batch_size : step * batch_size]]
        start = time.monotonic()
        loss_sum = kl_sum = 0.0
        tokens = 0
        for example in batch:
            input_ids, topk_ids, topk_logprobs = to_tensors(example, max_seq_len)
            loss, kl = student_loss(model, cartridge, input_ids.to(device), topk_ids.to(device), topk_logprobs.to(device))
            (loss / len(batch)).backward()
            loss_sum += loss.item()
            kl_sum += kl.item()
            tokens += input_ids.shape[1]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float("inf")).item()
        if not (math.isfinite(loss_sum) and math.isfinite(grad_norm)):
            raise FloatingPointError(f"step {step}: loss {loss_sum / len(batch)}, grad norm {grad_norm}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scale = warmup.get_last_lr()[0] / optimizer.param_groups[0]["initial_lr"]
        warmup.step()
        elapsed = time.monotonic() - start
        metrics = {
            "train/loss": loss_sum / len(batch),
            "train/kl": kl_sum / len(batch),
            "train/grad_norm": grad_norm,
            "train/lr": lr * scale,
            "time/step_s": elapsed,
            "throughput/step_tok_s": tokens / elapsed,
        }
        log(metrics, step)
        if on_step(step):
            break
    return metrics


def _epoch_order(n: int, epochs: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    order: list[int] = []
    for _ in range(epochs):
        idx = list(range(n))
        rng.shuffle(idx)
        order.extend(idx)
    return order


def weighted_order(groups: list[str], mix: dict[str, float], n: int, seed: int = 0) -> list[int]:
    """`n` example indices: each draw picks a group with probability mix[group]
    (with replacement), then the group's next example from its own reshuffled
    cycle, so a group's examples repeat evenly. `groups[i]` is example i's
    group (its task origin). Groups in `mix` must all have examples."""
    rng = random.Random(seed)
    members = {g: [i for i, x in enumerate(groups) if x == g] for g in mix}
    empty = [g for g, idx in members.items() if not idx]
    if empty:
        raise ValueError(f"no examples for {empty}")
    cycles: dict[str, list[int]] = {g: [] for g in mix}
    names, weights = list(mix), [mix[g] for g in mix]
    order = []
    for g in rng.choices(names, weights, k=n):
        if not cycles[g]:
            cycles[g] = rng.sample(members[g], len(members[g]))
        order.append(cycles[g].pop())
    return order


def effective_epochs(groups: list[str], order: list[int]) -> dict[str, float]:
    """Passes over each group's examples that `order` makes."""
    sizes = Counter(groups)
    drawn = Counter(groups[i] for i in order)
    return {g: drawn[g] / sizes[g] for g in sorted(sizes)}


# ---------------------------------------------------------------- Modal


class PipelineConfig(BaseModel):
    """Stage 1, then stage 2 (grpo.GRPOConfig fields; skipped if None). A
    heuristic cartridge eval on `eval_path` closes stage 1 (logged in the
    distill run); the final eval after the last stage is its own eval run,
    graded by the teacher if `judge`. All runs share distill.group."""

    model_config = ConfigDict(extra="forbid")

    distill: DistillConfig = DistillConfig()
    grpo: dict | None = {}
    qa_train_path: str = f"{VOLUME_PATH}/data/grpo/qa_train.jsonl"
    eval_path: str  # /vol/data/eval/<file>; required, no default eval set
    eval_max_items: int = 0  # checkpoint eval only; 0: all
    eval_max_new_tokens: int = 128
    judge: bool = True
    qa_eval: bool = True  # False: skip the qa_heldout checkpoint and final evals (e.g. a short probe)
    init_cartridge: str = ""  # a saved cartridge dir on the volume: skip stage 1, then GRPO (if set) and eval
    # Open-ended split='eval' tasks (Task Store): cartridge vs teacher (journal in
    # context) with tasks/judge.py, at every distill checkpoint (save_every) and
    # after the last stage.
    task_eval: bool = False
    task_final_n: int = 60  # stratified by origin (human has only 8)
    task_max_new_tokens: int = 3072  # final pass: the teacher's label budget, for both systems
    task_ckpt_n: int = 24  # 8 human / 8 chat / 8 psych_test
    task_ckpt_max_new_tokens: int = 768
    judge_rubric: str = "v0"


@cpu_app.function(cpu=4, memory=16384, volumes={VOLUME_PATH: volume}, secrets=secrets, timeout=30 * MINUTES)
def dry_run(config: PipelineConfig) -> dict:
    """Every step of `pipeline` before the model is needed, on CPU, with a real
    W&B run (role "dryrun") in the same group: tracking init, prepare_data
    (labels, shards, prompt check, origin order, lineage), the cartridge init
    prefix, and the eval/resume inputs the later stages read. A GPU launch
    can then only fail on the GPU work itself."""
    from transformers import AutoTokenizer

    from cartridges.cartridge.teacher import render

    volume.reload()
    d = config.distill
    tokenizer = AutoTokenizer.from_pretrained(d.model_id)
    with tracking.init_run("distill", d.model_id, d.group, config=d.model_dump(), p=d.p, role="dryrun") as run:
        examples, order, val = prepare_data(d, run, tokenizer)
        prefix_ids, _ = render(tokenizer, [{"role": "user", "content": "."}], Path(d.init_text_path).read_text())
        if len(prefix_ids) < d.p:
            raise ValueError(f"init prefix has {len(prefix_ids)} tokens < p={d.p}")
        needed = [p for p, used in ((config.eval_path, config.qa_eval), (config.qa_train_path, config.grpo is not None),
                                    (config.init_cartridge, bool(config.init_cartridge)),
                                    (d.resume_from, bool(d.resume_from))) if used]
        missing = [p for p in needed if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(missing)
        if d.resume_from:
            from cartridges.cartridge.cache import Cartridge

            resumed = Cartridge.load(d.resume_from)
            if resumed.p != d.p:
                raise ValueError(f"resume_from has p={resumed.p}, config p={d.p}")
            if Path(d.resume_from).resolve().parent == Path(d.out_dir).resolve():
                raise ValueError("out_dir would overwrite the checkpoint being resumed")
        if config.task_eval:
            from cartridges.cartridge.tasks import store
            from cartridges.cartridge.tasks.types import TaskQuery

            with store.connect() as conn:
                n_eval_tasks = len(store.fetch_tasks(conn, TaskQuery(split="eval", n=1, seed=0)))
        steps = math.ceil(len(order) / d.batch_size) if order is not None else None
        report = {"examples": len(examples), "steps": steps, "first_step": d.skip_samples // d.batch_size + 1, "val_examples": len(val.examples) if val else 0,
                  "train_tokens": sum(len(examples[i].prompt_ids) + len(examples[i].completion_ids) for i in order)
                  if order is not None else None,
                  "val_tokens": sum(len(e.prompt_ids) + len(e.completion_ids) for e in val.examples) if val else 0,
                  "lineage_missing": list(run.summary.get("lineage/missing", []) or []),
                  **({"eval_tasks": n_eval_tasks} if config.task_eval else {})}
        run.summary.update({f"dryrun/{k}": v for k, v in report.items() if not isinstance(v, list)})
    return report


@app.function(gpu=GPU, volumes={VOLUME_PATH: volume}, secrets=secrets, timeout=12 * 60 * MINUTES)
def pipeline(config: PipelineConfig) -> dict:
    """Load the base model once, then distill -> GRPO -> final eval. Each
    stage saves its cartridge under <out_dir>/<stage> on the volume and as a
    W&B artifact. Returns {stage: summary}."""
    from cartridges.cartridge.cache import Cartridge

    volume.reload()
    out = Path(config.distill.out_dir)
    results = {}
    tasks = TaskEval.start(config) if config.task_eval else None
    if config.init_cartridge:  # resume: skip stage 1
        model, tokenizer = _load_model(config.distill.model_id)
        cartridge = Cartridge.load(config.init_cartridge, device="cuda")
        prepare_frozen(model)
        ref = None
    else:
        model, tokenizer, cartridge, results["distill"] = _distill(config, out / "distill", tasks)
        ref = results["distill"]["artifact"]
    if config.grpo is not None:
        results["grpo"] = _grpo(model, tokenizer, cartridge, config, out / "grpo")
        ref = results["grpo"]["artifact"]
    if config.qa_eval:
        results["eval"] = _final_eval(model, tokenizer, cartridge, config, ref)
    if tasks:
        results["task_eval"] = tasks.final_run(model, tokenizer, cartridge, ref)
    return results


def _load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    volume.commit()  # keep the downloaded weights
    return model, tokenizer


def prepare_data(d: DistillConfig, run, tokenizer) -> tuple[list[Example], list[int] | None, LabeledSet | None]:
    """Everything stage 1 does before the model is needed: load and check the
    training (and validation) targets, the origin-weighted order, W&B lineage
    and data stats. Returns (examples, order, val). Runs on CPU, so
    `dry_run` exercises it before a GPU is paid for."""
    order = val = None
    if d.label_runs:
        train_set = _labeled_examples(d.label_runs, instructions_hash=d.instructions_hash, tokenizer=tokenizer)
        examples, origins = train_set.examples, train_set.origins
        if d.val_label_runs:
            val = _labeled_examples(d.val_label_runs, split="eval", instructions_hash=d.instructions_hash,
                                    tokenizer=tokenizer)
            val = pick_val(val, d.val_label_ids, d.val_max, d.seed)
            run.summary.update({"data/val_label_ids": [e.id for e in val.examples]})
            overlap = set(val.task_ids) & set(train_set.task_ids)
            if overlap:
                raise AssertionError(f"validation tasks also in training: {sorted(overlap)[:5]}")
        if d.origin_mix:
            order = weighted_order(origins, d.origin_mix, d.samples, d.seed)[d.skip_samples :]
            run.summary.update({"data/origin_counts": dict(Counter(origins)),
                                "data/effective_epochs": effective_epochs(origins, order)})
        for label_run in d.label_runs + d.val_label_runs:
            use_input(run, f"targets-{label_run}", "dataset")
    else:
        _use_data_artifacts(run, "selfstudy", "corpus")
        examples = load_examples(d.train_path)
    lengths = sorted(len(e.prompt_ids) + len(e.completion_ids) for e in examples)
    run.summary.update({
        "data/examples": len(examples),
        "data/seq_len_p50": lengths[len(lengths) // 2],
        "data/seq_len_p95": lengths[int(0.95 * (len(lengths) - 1))],
        "data/seq_len_max": lengths[-1],
        "data/truncated_examples": sum(n > d.max_seq_len for n in lengths),
    })
    return examples, order, val


def use_input(run, ref: str, type: str) -> bool:
    """Record a W&B artifact as this run's input. Lineage never stops training:
    a missing artifact (e.g. a label run that logged none) is a warning, and
    the ref is listed under lineage/missing in the summary (the run ids are in
    the config either way)."""
    try:
        run.use_artifact(ref, type=type, download=False)
        return True
    except Exception as e:  # noqa: BLE001 - wandb raises CommError and others
        logger.warning("W&B input %s not recorded: %s", ref, e)
        missing = list(run.summary.get("lineage/missing", []) or [])
        run.summary.update({"lineage/missing": [*missing, ref]})
        return False


def _distill(config: PipelineConfig, out: Path, tasks: "TaskEval | None" = None):
    """Stage 1 in one distill run (opened before the model loads, so its cost
    covers the load), ending with a heuristic checkpoint eval."""
    import torch

    from cartridges.cartridge.cache import Cartridge
    from cartridges.cartridge.evals import evaluate, log_report
    from cartridges.cartridge.teacher import render

    d = config.distill
    with tracking.init_run("distill", d.model_id, d.group, config=d.model_dump(), p=d.p,
                           role="resume" if d.resume_from else None) as run:
        torch.manual_seed(d.seed)
        with run.timer("load_model"):
            model, tokenizer = _load_model(d.model_id)
        examples, order, val = prepare_data(d, run, tokenizer)
        # The paper's init, on the teacher's own prefix: the first p tokens of
        # its context block (system header + instructions + start of the corpus).
        if d.resume_from:
            cartridge = Cartridge.load(d.resume_from, device="cuda")
            cartridge.meta.update({"resumed_from": d.resume_from, "distill_run_id": run.id})
        else:
            prefix_ids, _ = render(tokenizer, [{"role": "user", "content": "."}], Path(d.init_text_path).read_text())
            cartridge = Cartridge.from_prefill(
                model, torch.tensor([prefix_ids[: d.p]]), meta={"group": d.group, "distill_run_id": run.id}
            )
        prepare_frozen(model)  # after the init prefill: that runs in eval mode, without checkpointing

        offset = d.skip_samples // d.batch_size  # logged step = offset + step of this run
        best_dir = out.parent / "distill-best"

        def save_best() -> None:
            cartridge.save(best_dir)
            volume.commit()  # a crash keeps the best checkpoint

        best = EarlyStopping(d.patience, save=save_best, best=d.best_val_kl, step=offset)

        def validate(step: int, update: bool = True) -> bool:
            """Log val KL; keep the best checkpoint; True when patience runs out."""
            metrics = val_kl(model, cartridge, val, d.max_seq_len)
            stop = best.update(metrics["val/kl"], step) if update else False
            run.log({**metrics, "val/step": step, "val/best_kl": best.best, "val/best_step": best.step}, step=step)
            return stop

        if val:
            validate(offset, update=d.best_val_kl is None)  # a resumed checkpoint's own score does not count

        def log(metrics: dict, step: int) -> None:
            run.log(metrics, step=offset + step)

        def on_step(local_step: int) -> bool:
            step = offset + local_step
            if step == 1 or step % 10 == 0:
                run.log_gpu_memory(step=step)
            stop = bool(val) and step % d.val_every == 0 and validate(step)
            if d.save_every and step % d.save_every == 0:
                cartridge.save(out.parent / f"distill-step-{step}")
                volume.commit()
                if tasks:  # overfitting check per origin: judge/<stratum>/non_loss over steps
                    summary = tasks.judge(model, tokenizer, cartridge, "ckpt")[1]
                    run.log({f"ckpt_{k}": v for k, v in judge_scalars(summary).items()}, step=step)
            return stop

        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        final = train(
            model, cartridge, examples,
            lr=d.lr, batch_size=d.batch_size, epochs=d.epochs, max_steps=d.max_steps,
            max_seq_len=d.max_seq_len, seed=d.seed, relative_lr=d.relative_lr, order=order,
            warmup_steps=d.warmup_steps, log=log, on_step=on_step,
        )
        if val and d.patience:
            # End with the best checkpoint: this run's, or the resumed one if nothing beat it.
            cartridge = Cartridge.load(best_dir if best.saved else d.resume_from or best_dir, device="cuda")
            reason = "patience" if best.bad >= d.patience else "max_steps"
            # W&B steps only go forward, so the marker is logged after the last step and carries its own x.
            run.log({"early_stop/val_kl": best.best, "early_stop/at_step": best.step, "val/step": best.step})
            run.summary.update({"early_stop/step": best.step, "early_stop/best_val_kl": best.best,
                                "early_stop/reason": reason})
        cartridge.save(out)
        volume.commit()
        summary = {
            **final,
            **({"best_step": best.step, "best_val_kl": best.best} if val else {}),
            "examples": len(examples),
            "completion_tokens": sum(len(e.completion_ids) for e in examples),
            "train_s": time.monotonic() - start,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
            "cartridge_path": str(out),
            "cache_mem_gb": cartridge.nbytes() / 1e9,
            "wandb_run_id": run.id,
            "artifact": run.log_artifact(out, type="cartridge", metadata={"p": d.p, "model": d.model_id}),
        }
        if config.qa_eval:
            items, pages = _eval_data(config)
            report = evaluate("cartridge", items, _answer_fn(model, tokenizer, cartridge, config), pages, None,
                              stage="distill")
            log_report(run, report, step=None)
            summary["checkpoint_eval"] = {k: v for k, v in report.items() if k != "rows"}
        run.summary.update({k: v for k, v in summary.items() if k != "checkpoint_eval"})
    return model, tokenizer, cartridge, summary


def _grpo(model, tokenizer, cartridge, config: PipelineConfig, out: Path) -> dict:
    from cartridges.cartridge.evals import grounded_reward, load_pages, read_jsonl
    from cartridges.cartridge.grpo import GRPOConfig, train_grpo

    d = config.distill
    g = GRPOConfig(**config.grpo)
    items = read_jsonl(Path(config.qa_train_path))
    pages = load_pages(Path(JOURNALS_PATH))
    with tracking.init_run("grpo", d.model_id, d.group, config=g.model_dump(), p=cartridge.p) as run:
        cartridge.meta["grpo_run_id"] = run.id
        _use_data_artifacts(run, "qa-train")

        def save_every_25(step: int) -> None:  # a timeout keeps the stage's progress
            if step % 25 == 0:
                cartridge.save(out.parent / f"grpo-step-{step}")
                volume.commit()
        summary = train_grpo(
            model, tokenizer, cartridge, items,
            reward_fn=lambda text, item: grounded_reward(text, item, pages),
            config=g, log=run.log, on_step=save_every_25,
        )
        cartridge.save(out)
        volume.commit()
        summary = {**summary, "items": len(items), "cartridge_path": str(out), "wandb_run_id": run.id,
                   "artifact": run.log_artifact(out, type="cartridge", metadata={"p": cartridge.p, "model": d.model_id})}
        run.summary.update(summary)
    return summary


def _final_eval(model, tokenizer, cartridge, config: PipelineConfig, cartridge_ref: str | None) -> dict:
    """The cartridge row of the 3-way comparison (eval runs base and teacher):
    every item, teacher judge, median TTFT over 5 single requests."""
    import statistics

    from cartridges.cartridge.evals import question_messages, read_jsonl, run_eval, teacher_system

    d = config.distill
    judge = None
    if config.judge:
        complete, _ = teacher_system(d.model_id, with_context=False, max_tokens=8)
        judge = lambda messages: complete(messages).text  # noqa: E731
    question = read_jsonl(Path(config.eval_path))[0]["question"]
    messages = question_messages(question)
    _ttft(model, tokenizer, cartridge, messages)  # warm-up: fla's triton kernels JIT/autotune on first call
    ttft = statistics.median(_ttft(model, tokenizer, cartridge, messages) for _ in range(5))
    report = run_eval(
        "cartridge", _answer_fn(model, tokenizer, cartridge, config), d.group, d.model_id,
        items_path=Path(config.eval_path), out_dir=Path(JOURNALS_PATH), judge=judge, p=cartridge.p, gpu=tracking.detect_gpu(),
        cartridge_ref=cartridge_ref, ttft_s=ttft, cache_mem_gb=cartridge.nbytes() / 1e9,
    )
    return {k: v for k, v in report.items() if k != "rows"}


class LabeledSet(BaseModel):
    """Teacher targets of one split, with each example's task origin and id."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    examples: list[Example]
    origins: list[str]
    task_ids: list[str]


def _labeled_examples(label_runs: list[str], split: str = "train", instructions_hash: str = "",
                      tokenizer=None) -> LabeledSet:
    """The Teacher's targets for the validated labels of `label_runs` on tasks
    of `split` ("train" for training, "eval" for the validation KL)."""
    from cartridges.cartridge.tasks import label, store, targets
    from cartridges.cartridge.tasks.types import TaskQuery

    with store.connect() as conn:
        labels = store.fetch_labels(conn, label_runs, split=split)  # `split` tasks only, by construction
        tasks = {s: store.fetch_tasks(conn, TaskQuery(split=s, n=1, seed=0)) for s in ("train", "eval", "exemplar")}
    labels = current_labels(labels, instructions_hash or label.instructions_hash())
    origins = label_origins(labels, tasks, split)
    examples = list(targets.read_labeled(labels, root=VOLUME_PATH))  # stored /cache/... or relative -> /vol/...
    if not examples:
        raise ValueError(f"no validated {split} labels with targets in runs {label_runs}")
    if tokenizer is not None:
        text = {t.task_id: t.text for t in tasks[split]}
        check_prompts(examples, [text[lab.task_id] for lab in labels], tokenizer)
    return LabeledSet(examples=examples, origins=origins, task_ids=[str(lab.task_id) for lab in labels])


def check_prompts(examples: list[Example], task_texts: list[str], tokenizer, n: int = 20) -> None:
    """The labels' prompt_ids (the student's input) must be the task rendered
    the way students and evals render it: label.answer_messages (citation
    reminder + task) through teacher.render. Checks the first `n`."""
    from cartridges.cartridge.tasks.label import answer_messages

    bad = [ex.id for ex, text in list(zip(examples, task_texts, strict=True))[:n]
           if ex.prompt_ids != _prompt_ids(tokenizer, answer_messages(text))]
    if bad:
        raise ValueError(f"label prompt_ids differ from answer_messages rendering: {bad[:5]}")


def current_labels(labels: list, instructions_hash: str) -> list:
    """Validated labels with targets made under `instructions_hash`; raises if
    none are (e.g. a stale run id), and logs how many other-hash labels it dropped."""
    usable = [lab for lab in labels if lab.status == "validated" and lab.targets_uri is not None]
    kept = [lab for lab in usable if lab.instructions_hash == instructions_hash]
    if usable and not kept:
        raise ValueError(f"no labels with instructions_hash {instructions_hash}; found {sorted({l.instructions_hash for l in usable})}")
    if len(kept) < len(usable):
        logger.warning("dropped %d labels with another instructions_hash", len(usable) - len(kept))
    return kept


def label_origins(labels: list, tasks: dict[str, list], split: str = "train") -> list[str]:
    """Each label's task origin. Raises if a label's task is in another split,
    or is not a validated task of `split`."""
    other = {t.task_id for s, ts in tasks.items() if s != split for t in ts}
    leaked = [str(lab.label_id) for lab in labels if lab.task_id in other]
    if leaked:
        raise AssertionError(f"{split} labels on tasks of another split: {leaked[:5]}")
    origin = {t.task_id: t.origin for t in tasks[split]}
    missing = [str(lab.label_id) for lab in labels if lab.task_id not in origin]
    if missing:
        raise ValueError(f"labels whose task is not a validated {split} task: {missing[:5]}")
    return [origin[lab.task_id] for lab in labels]


def val_kl(model, cartridge, val: LabeledSet, max_seq_len: int) -> dict:
    """Teacher-forced KL(teacher top-k || student) on held-out labeled tasks,
    mean over examples, overall and per origin. No decoding, no gradients."""
    import torch

    device = next(cartridge.parameters()).device
    kls = []
    with torch.no_grad():
        for ex in val.examples:
            ids, top, lp = to_tensors(ex, max_seq_len)
            kls.append(student_loss(model, cartridge, ids.to(device), top.to(device), lp.to(device))[1].item())
    out = {"val/kl": sum(kls) / len(kls)}
    for origin in sorted(set(val.origins)):
        vals = [k for k, o in zip(kls, val.origins) if o == origin]
        out[f"val/kl/{origin}"] = sum(vals) / len(vals)
    return out


class EarlyStopping:
    """Tracks the best validation KL. update() calls `save` on a new best (when
    patience > 0, so there is a checkpoint to end with) and returns True once
    `patience` evals in a row bring no improvement."""

    def __init__(self, patience: int, save: Callable[[], object], best: float | None = None, step: int = 0):
        self.patience, self.save = patience, save
        self.best, self.step, self.bad = float("inf") if best is None else best, step, 0
        self.saved = False

    def update(self, kl: float, step: int) -> bool:
        if kl < self.best:
            self.best, self.step, self.bad = kl, step, 0
            if self.patience:
                self.save()
                self.saved = True
        else:
            self.bad += 1
        return bool(self.patience) and self.bad >= self.patience


def pick_val(val: LabeledSet, label_ids: list[str], n: int, seed: int) -> LabeledSet:
    """The validation set: exactly `label_ids` if given (raises if any is no
    longer a validated label), else a stratified subset of `n` (0: all)."""
    if label_ids:
        index = {e.id: i for i, e in enumerate(val.examples)}
        missing = [x for x in label_ids if x not in index]
        if missing:
            raise ValueError(f"{len(missing)} val labels not found (relabelled or rejected?): {missing[:5]}")
        idx = [index[x] for x in label_ids]
        return LabeledSet(examples=[val.examples[i] for i in idx], origins=[val.origins[i] for i in idx],
                          task_ids=[val.task_ids[i] for i in idx])
    return subset_labeled(val, n, seed) if n else val


def subset_labeled(labeled: LabeledSet, n: int, seed: int = 0) -> LabeledSet:
    """A fixed origin-stratified subset of `n` labeled examples (stratified_subset)."""
    from types import SimpleNamespace

    keys = [SimpleNamespace(task_id=f"{tid}:{i}", origin=o, i=i)
            for i, (tid, o) in enumerate(zip(labeled.task_ids, labeled.origins, strict=True))]
    idx = sorted(k.i for k in stratified_subset(keys, n, seed))
    return LabeledSet(examples=[labeled.examples[i] for i in idx], origins=[labeled.origins[i] for i in idx],
                      task_ids=[labeled.task_ids[i] for i in idx])


def stratified_subset(tasks: list, n: int, seed: int = 0) -> list:
    """`n` tasks, as equal a share per origin as there are tasks for: an
    origin with fewer than its share gives all it has and the rest is split
    among the others. Deterministic for a seed; sorted by task_id."""
    rng = random.Random(seed)
    by_origin: dict[str, list] = {}
    for t in sorted(tasks, key=lambda t: str(t.task_id)):
        by_origin.setdefault(t.origin, []).append(t)
    quota = dict.fromkeys(by_origin, 0)
    left = min(n, len(tasks))
    while left:
        open_ = [o for o in sorted(by_origin) if quota[o] < len(by_origin[o])]
        share = max(1, left // len(open_))
        for o in open_:
            take = min(share, len(by_origin[o]) - quota[o], left)
            quota[o] += take
            left -= take
            if not left:
                break
    picked = [t for o in sorted(by_origin) for t in rng.sample(by_origin[o], quota[o])]
    return sorted(picked, key=lambda t: str(t.task_id))


def _use_data_artifacts(run, *kinds: str) -> None:
    """Record the datasets this run reads from the volume as its W&B inputs
    (lineage only; the files themselves come from the volume). Refs come from
    data/manifest.json, written by data.build_datasets."""
    manifest = Path(VOLUME_PATH) / "data" / "manifest.json"
    if not manifest.exists():
        logger.warning("no %s: W&B lineage for %s not recorded", manifest, kinds)
        return
    refs = json.loads(manifest.read_text())
    for kind in kinds:
        use_input(run, refs[kind], "dataset")


def task_judge(answer, tasks: list, teacher_answers: list[str], generate, rubric) -> tuple[list, dict, list[dict]]:
    """Cartridge (X) vs teacher (Y) on open-ended tasks: `answer(conversations)`
    gives the cartridge's answers (answer_messages rows), `generate` is the
    judge (data.modal_generate shape). Returns (outcomes, summary, answers)."""
    from cartridges.cartridge.tasks.judge import Item, judge_pairs, summarize
    from cartridges.cartridge.tasks.label import task_messages

    answers = answer([task_messages(t) for t in tasks])
    items = [Item(task_id=str(t.task_id), origin=t.origin, kind=t.kind, question=t.text, x=a["text"], y=y)
             for t, a, y in zip(tasks, answers, teacher_answers, strict=True)]
    outcomes = judge_pairs(items, generate, rubric)
    return outcomes, summarize(outcomes), answers


def judge_scalars(summary: dict[str, dict]) -> dict:
    """judge/<stratum>/<stat> scalars, as tasks.judge.log_comparison logs them."""
    return {f"judge/{stratum}/{k}": v for stratum, stats in summary.items() for k, v in stats.items()
            if isinstance(v, (int, float))}


class TaskEval:
    """Judge checks on fixed stratified subsets of the split='eval' tasks (by
    origin, seeded): `ckpt` (task_ckpt_n, answers capped at
    task_ckpt_max_new_tokens) at every distill checkpoint, and `final`
    (task_final_n, task_max_new_tokens) after the last stage; ckpt is a subset
    of final. The teacher's answers to both are requested when the pipeline
    starts, so they are ready by the first checkpoint."""

    def __init__(self, config: PipelineConfig, subsets: dict, teacher: dict, generate, rubric):
        self.config, self.subsets, self._teacher, self.generate, self.rubric = config, subsets, teacher, generate, rubric

    @classmethod
    def start(cls, config: PipelineConfig) -> "TaskEval":
        from concurrent.futures import ThreadPoolExecutor

        from cartridges.cartridge.data import modal_generate
        from cartridges.cartridge.tasks import store
        from cartridges.cartridge.tasks.judge import load_rubric
        from cartridges.cartridge.tasks.label import task_messages
        from cartridges.cartridge.tasks.types import TaskQuery
        from cartridges.cartridge.teacher import teacher_cls

        with store.connect() as conn:
            tasks = store.fetch_tasks(conn, TaskQuery(split="eval", n=1, seed=0))
        final = stratified_subset(tasks, config.task_final_n, config.distill.seed)
        subsets = {"final": final, "ckpt": stratified_subset(final, config.task_ckpt_n, config.distill.seed)}
        budget = {"final": config.task_max_new_tokens, "ckpt": config.task_ckpt_max_new_tokens}
        generate = modal_generate(teacher_cls("Teacher", config.distill.model_id))
        pool = ThreadPoolExecutor(2)
        teacher = {name: pool.submit(generate, [task_messages(t) for t in subset], with_context=True,
                                     temperature=0.0, max_tokens=budget[name])
                   for name, subset in subsets.items()}
        return cls(config, subsets, teacher, generate, load_rubric(config.judge_rubric))

    def judge(self, model, tokenizer, cartridge, which: str = "ckpt"):
        budget = self.config.task_ckpt_max_new_tokens if which == "ckpt" else self.config.task_max_new_tokens
        eos = _eos_ids(tokenizer)
        answer = lambda convs: answer_messages(model, tokenizer, cartridge, convs, budget, eos)  # noqa: E731
        teacher = [c.text if c is not None else "" for c in self._teacher[which].result()]
        return task_judge(answer, self.subsets[which], teacher, self.generate, self.rubric)

    def final_run(self, model, tokenizer, cartridge, cartridge_ref: str | None) -> dict:
        """The cartridge-vs-teacher comparison on the final subset, as its own eval run."""
        from cartridges.cartridge.tasks.judge import judge_config, log_comparison

        d = self.config.distill
        config = judge_config(self.rubric, d.model_id, x="cartridge", y="teacher") | {
            "system": "cartridge", "eval": "tasks", "n_tasks": len(self.subsets["final"]),
            "task_ids": [str(t.task_id) for t in self.subsets["final"]]}
        with tracking.init_run("eval", d.model_id, d.group, config, p=cartridge.p, role="tasks") as run:
            if cartridge_ref:
                use_input(run, cartridge_ref, "cartridge")
            outcomes, summary, answers = self.judge(model, tokenizer, cartridge, "final")
            log_comparison(run, outcomes, summary)
            lengths = {"tasks/truncation_rate": sum(a["finish_reason"] == "length" for a in answers) / max(len(answers), 1),
                       "tasks/mean_tokens": sum(a["n_tokens"] for a in answers) / max(len(answers), 1)}
            run.log(lengths)
            run.summary.update(lengths)
        return {**judge_scalars(summary), **lengths}


def _eval_data(config: PipelineConfig):
    from cartridges.cartridge.evals import load_pages, read_jsonl

    items = read_jsonl(Path(config.eval_path))
    return items[: config.eval_max_items] if config.eval_max_items else items, load_pages(Path(JOURNALS_PATH))


def _prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Tokenized like every other stage (teacher.render, non-thinking)."""
    from cartridges.cartridge.teacher import render

    return render(tokenizer, messages)[1]


def _eos_ids(tokenizer) -> list[int]:
    return [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]


def answer_messages(model, tokenizer, cartridge, conversations: list[list[dict]], max_new_tokens: int,
                    eos_ids: list[int]) -> list[dict]:
    """Greedy answers with the cartridge, one conversation at a time
    (Cartridge.generate takes unpadded prompts), tokenized by teacher.render.
    Each is {"text", "finish_reason" ("stop" | "length"), "n_tokens"}."""
    import torch

    out = []
    for messages in conversations:
        ids = torch.tensor([_prompt_ids(tokenizer, messages)], device=model.device)
        new = cartridge.generate(model, ids, max_new_tokens=max_new_tokens, eos_token_id=eos_ids)[0].tolist()
        stop = next((i for i, t in enumerate(new) if t in eos_ids), None)
        out.append({
            "text": tokenizer.decode(new[:stop], skip_special_tokens=True),
            "finish_reason": "stop" if stop is not None else "length",
            "n_tokens": stop + 1 if stop is not None else len(new),
        })
    return out


def _answer_fn(model, tokenizer, cartridge, config: PipelineConfig):
    """evals' answer(batch) on the cartridge, greedy."""
    eos = _eos_ids(tokenizer)

    def answer(batch: list[list[dict]]) -> list[str]:
        return [r["text"] for r in answer_messages(model, tokenizer, cartridge, batch, config.eval_max_new_tokens, eos)]

    return answer


def _ttft(model, tokenizer, cartridge, messages: list[dict]) -> float:
    """Seconds from the request to the first token's logits (prompt prefill on the cartridge)."""
    import torch

    ids = torch.tensor([_prompt_ids(tokenizer, messages)], device=model.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        cartridge(model, ids, logits_to_keep=1).logits.argmax(-1)
    torch.cuda.synchronize()
    return time.perf_counter() - start


class SweepConfig(BaseModel):
    """Short stage-1 runs on one loaded model, one distill run each (role
    sweep-p<p>-lr<lr>), scored by validation KL only (no judge, no saves)."""

    model_config = ConfigDict(extra="forbid")

    base: DistillConfig  # label_runs, val_label_runs, origin_mix, group, model_id, ...
    lrs: list[float] = [1e-2, 2e-2, 5e-2]  # relative lr (base.relative_lr)
    ps: list[int] = [512, 2048]
    steps: int = 30


@app.function(gpu=GPU, volumes={VOLUME_PATH: volume}, secrets=secrets, timeout=6 * 60 * MINUTES)
def sweep(config: SweepConfig) -> dict:
    """{"p<p>-lr<lr>": [val metrics at step 0, val_every, ..., steps]}."""
    import torch

    from cartridges.cartridge.cache import Cartridge
    from cartridges.cartridge.teacher import render

    b = config.base
    if not (b.label_runs and b.val_label_runs):
        raise ValueError("sweep needs label_runs and val_label_runs")
    volume.reload()
    model, tokenizer = _load_model(b.model_id)
    train_set = _labeled_examples(b.label_runs, instructions_hash=b.instructions_hash, tokenizer=tokenizer)
    val = _labeled_examples(b.val_label_runs, split="eval", instructions_hash=b.instructions_hash, tokenizer=tokenizer)
    val = pick_val(val, b.val_label_ids, b.val_max, b.seed)
    if set(val.task_ids) & set(train_set.task_ids):
        raise AssertionError("validation tasks also in training")
    order = weighted_order(train_set.origins, b.origin_mix, config.steps * b.batch_size, b.seed) if b.origin_mix else None
    prefix_ids, _ = render(tokenizer, [{"role": "user", "content": "."}], Path(b.init_text_path).read_text())
    prepare_frozen(model)
    results = {}
    for p in config.ps:
        for lr in config.lrs:
            d = b.model_copy(update={"p": p, "lr": lr, "max_steps": config.steps, "samples": config.steps * b.batch_size})
            name = f"p{p}-lr{lr:g}"
            with tracking.init_run("distill", d.model_id, d.group, config=d.model_dump() | {"sweep": name}, p=p,
                                   role=f"sweep-{name}") as run:
                model.eval()  # the init prefill runs without checkpointing
                cartridge = Cartridge.from_prefill(model, torch.tensor([prefix_ids[:p]]))
                model.train()
                curve = [val_kl(model, cartridge, val, d.max_seq_len)]
                run.log({**curve[0], "val/step": 0}, step=0)

                def on_step(step: int) -> None:
                    if step % d.val_every == 0 or step == config.steps:
                        curve.append(val_kl(model, cartridge, val, d.max_seq_len))
                        run.log({**curve[-1], "val/step": step}, step=step)

                train(model, cartridge, train_set.examples, lr=lr, batch_size=d.batch_size, max_steps=config.steps,
                      max_seq_len=d.max_seq_len, seed=d.seed, relative_lr=d.relative_lr, order=order,
                      warmup_steps=d.warmup_steps, log=run.log, on_step=on_step)
                run.summary.update({"sweep/final_val_kl": curve[-1]["val/kl"]})
            results[name] = curve
            del cartridge
            torch.cuda.empty_cache()
    return results


def score_examples(model, cartridge, examples: list[Example], task_ids: list[str] | None = None,
                   max_seq_len: int = 3584, chunk: int = 1024) -> list[dict]:
    """Teacher-forced per-token scores of the teacher's answers. Per example:
    {"id" (label_id), "task_id", "n_tokens", "logprobs": the model's fp32 logprob
    of each answer token, "argmax": its top-1 token id there, "teacher_logprobs": the teacher's (None outside its
    top-k), "kl_topk": mean top-k KL}. `cartridge=None` scores the base model
    alone (no prefix, positions from 0). A sequence longer than max_seq_len
    scores the answer prefix that fits (n_tokens says how much)."""
    import torch

    device = next(model.parameters()).device
    rows = []
    with torch.no_grad():
        for i, ex in enumerate(examples):
            ids, top, lp = (t.to(device) for t in to_tensors(ex, max_seq_len))
            T, L = top.shape[0], ids.shape[1]
            p = cartridge.p if cartridge is not None else 0
            hidden = model.model(
                input_ids=ids, position_ids=torch.arange(p, p + L, device=device)[None], use_cache=False,
                past_key_values=cartridge.as_past_key_values(1) if cartridge is not None else None,
            ).last_hidden_state[0, L - T - 1 : L - 1]  # position i predicts token i+1
            target = ids[0, L - T :]
            student, argmax, kl = [], [], 0.0
            for a in range(0, T, chunk):
                logq = torch.log_softmax(model.lm_head(hidden[a : a + chunk]).float(), dim=-1)
                student += logq.gather(-1, target[a : a + chunk, None])[:, 0].tolist()
                argmax += logq.argmax(-1).tolist()
                q_top, p_top = logq.gather(-1, top[a : a + chunk]), lp[a : a + chunk]
                kl += (p_top.exp() * (p_top - q_top)).sum().item()
            teacher = [lp[t][top[t] == target[t]].tolist() for t in range(T)]
            rows.append({"id": ex.id, "task_id": task_ids[i] if task_ids else None, "n_tokens": T,
                         "logprobs": student, "argmax": argmax,
                         "teacher_logprobs": [x[0] if x else None for x in teacher],
                         "kl_topk": kl / T})
    return rows


@app.function(gpu=GPU, volumes={VOLUME_PATH: volume}, secrets=secrets, timeout=6 * 60 * MINUTES)
def score_tasks(cartridge: str | None, labels: list[dict], model_id: str = "Qwen/Qwen3.5-4B",
                with_base: bool = True) -> dict:
    """For eval's CGC: score_examples of labeled tasks' teacher answers under
    the cartridge and, if `with_base`, the base model alone, in one HF forward
    path. `labels` are tasks.types.Label dicts (Label.model_dump(mode="json"),
    e.g. store.fetch_labels(..., split="eval")); targets come from the Target
    Store. `cartridge`: a /vol/... dir or cartridge-<run_id>:latest; None
    (or "") scores the base model only, on the same path and prompt_ids.
    Returns {"cartridge": rows, "base": rows} (only "base" without a cartridge)."""
    from cartridges.cartridge.cache import Cartridge
    from cartridges.cartridge.tasks import targets
    from cartridges.cartridge.tasks.types import Label

    volume.reload()
    labels = [lab for lab in (Label.model_validate(x) for x in labels)
              if lab.status == "validated" and lab.targets_uri is not None]
    examples = list(targets.read_labeled(labels, root=VOLUME_PATH))  # stored /cache/... or relative -> /vol/...
    task_ids = [str(lab.task_id) for lab in labels]
    model, _ = _load_model(model_id)
    cart = None
    if cartridge:
        path = cartridge if cartridge.startswith("/") else tracking.fetch_artifact(cartridge, type="cartridge")
        cart = Cartridge.load(path, device="cuda")
    return score_systems(model, cart, examples, task_ids, with_base)


def score_systems(model, cartridge, examples: list[Example], task_ids: list[str], with_base: bool = True) -> dict:
    """{"cartridge": rows, "base": rows} on one HF path; no cartridge -> base only."""
    out = {"cartridge": score_examples(model, cartridge, examples, task_ids)} if cartridge is not None else {}
    if with_base or cartridge is None:
        out["base"] = score_examples(model, None, examples, task_ids)
    return out


def stream_answers(answer_one: Callable[[str], dict], tasks: list[dict], out_path: Path,
                   commit: Callable[[], None] = lambda: None, every: int = 5,
                   log: Callable[[str], None] = logger.info):
    """Yield one answer row per task ({"task_id", "text", "finish_reason",
    "n_tokens"}) as soon as it is decoded. Each new row is appended to the
    JSONL `out_path` first (commit() every `every` rows and at the end), and a
    progress line (done/total, tok/s, ETA) is logged every `every` rows. Tasks
    already in `out_path` (an earlier, interrupted call) are yielded from it,
    not decoded again. `tasks`: [{"task_id", "text"}]."""
    done = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["task_id"]] = row
    out_path.parent.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if t["task_id"] not in done]
    log(f"answers: {len(tasks) - len(todo)} of {len(tasks)} already in {out_path}")
    for t in tasks:
        if t["task_id"] in done:
            yield done[t["task_id"]]
    start, tokens = time.monotonic(), 0
    with open(out_path, "a") as f:
        for i, t in enumerate(todo, 1):
            row = {"task_id": t["task_id"], **answer_one(t["text"])}
            f.write(json.dumps(row) + "\n")
            f.flush()
            tokens += row["n_tokens"]
            if i % every == 0 or i == len(todo):
                commit()
                elapsed = time.monotonic() - start
                log(f"answers: {i}/{len(todo)} new, {tokens / elapsed:.1f} tok/s, "
                    f"ETA {(len(todo) - i) * elapsed / i / 60:.1f} min")
            yield row


@app.function(gpu=GPU, volumes={VOLUME_PATH: volume}, secrets=secrets, timeout=6 * 60 * MINUTES)
def answer_tasks(cartridge: str, tasks: list[dict], run_name: str, model_id: str = "Qwen/Qwen3.5-4B",
                 max_new_tokens: int = 1536):
    """Generator (call with .remote_gen): the cartridge's answers to open-ended
    tasks for eval's judge, in the teacher's task format (label.answer_messages:
    citation reminder + task), greedy, one row per task as soon as it is
    decoded. `tasks`: [{"task_id", "text"}]. Rows are also appended to
    /vol/answers/<run_name>.jsonl and committed every 5, so they survive a
    client disconnect, and a rerun with the same run_name resumes (finished
    tasks are yielded from the file). `cartridge`: a /vol/... dir or a W&B ref
    (cartridge-<run_id>:latest). Row: {"task_id", "text", "finish_reason", "n_tokens"}."""
    from cartridges.cartridge.cache import Cartridge
    from cartridges.cartridge.tasks.label import answer_messages as task_turns

    volume.reload()
    path = cartridge if cartridge.startswith("/") else tracking.fetch_artifact(cartridge, type="cartridge")
    model, tokenizer = _load_model(model_id)
    cart = Cartridge.load(path, device="cuda")
    eos = _eos_ids(tokenizer)

    def answer_one(text: str) -> dict:  # as the teacher labels the task
        return answer_messages(model, tokenizer, cart, [task_turns(text)], max_new_tokens, eos)[0]

    yield from stream_answers(answer_one, tasks, Path(VOLUME_PATH) / "answers" / f"{run_name}.jsonl", volume.commit)


@app.local_entrypoint()
def main(eval_set: str, max_steps: int = 0, p: int = 2048, group: str = "", grpo_steps: int = 0) -> None:
    group = group or tracking.group_name("distill")
    config = PipelineConfig(
        distill=DistillConfig(max_steps=max_steps, p=p, group=group, out_dir=f"{VOLUME_PATH}/cartridges/{group}"),
        grpo={"max_steps": grpo_steps} if grpo_steps else None,
        eval_path=f"{VOLUME_PATH}/data/eval/{eval_set}",
    )
    tracking.check_budget(tracking.GPU_RATES_USD_PER_HOUR[GPU] * 2)  # <= 2 h on the A100 + judge calls
    print(json.dumps(pipeline.remote(config), indent=2, default=str))
