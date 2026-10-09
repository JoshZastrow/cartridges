"""Profile one fine-tune training step (forward + backward + optimizer) on Modal.

`train-step` mode of the single-page profiler (docs/plans/profiling.md, step 15).
Builds one batch with the same collator, LoRA setup, autocast and 8-bit AdamW
the Trainer uses (`finetune.py::_train`), runs untraced warm-up steps, then
records one step with torch.profiler and CUDA memory history. Nothing in the
training path changes; this only calls its building blocks.

Run:
uv run modal run -m cartridges.training.profiling_train::run_profile_train \\
    --run-name prof-train-bs1 --batch-size 1
uv run modal volume get cartridges-training-cache profiling/prof-train-bs1 profiles/
"""

from __future__ import annotations

import time

import modal

from cartridges.training.config import TrainConfig
from cartridges.training.finetune import (
    MINUTES,
    WANDB_ENTITY,
    WANDB_PROJECT,
    _load_base_and_lora,
    _stage,
    image,
    logger,
    model_cache,
    wandb_secret,
)

PROFILE_ROOT = "/cache/profiling"
PROFILE_GROUP = "infra-profiling"

# CUDA runtime calls that launch work, and ones that block the CPU until the
# GPU catches up (a host-device sync). Counted from the trace's CPU events.
_LAUNCH_CALLS = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC")
_SYNC_CALLS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")
_COPY_CALLS = ("cudaMemcpyAsync", "cudaMemcpy")

app = modal.App("cartridges-profiling-train", image=image)


def count_cuda_api_calls(event_names: list[str]) -> dict[str, int]:
    """Count kernel launches, explicit syncs and memcpy calls by CUDA runtime
    API name. (A device-to-host memcpy also blocks the host; the trace can't
    tell copy direction from the name alone, so copies are reported apart.)"""
    return {
        "launches": sum(name in _LAUNCH_CALLS for name in event_names),
        "syncs": sum(name in _SYNC_CALLS for name in event_names),
        "memcpys": sum(name in _COPY_CALLS for name in event_names),
    }


def _to_device(obj, device: str):
    """Recursively move tensors in nested dicts/lists/tuples, the way
    Trainer._prepare_inputs does for the collator's output."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


@app.function(
    gpu="A100-80GB",  # same GPU as finetune(), so memory numbers transfer
    timeout=30 * MINUTES,
    volumes={"/cache": model_cache},
    secrets=[wandb_secret],
)
def profile_train_step(
    run_name: str,
    config: TrainConfig = TrainConfig(),
    batch_size: int = 1,
    index: int = 0,
    warmup_steps: int = 2,
    with_stack: bool = True,
) -> dict:
    import gzip
    import json
    import shutil
    import subprocess
    from pathlib import Path

    import bitsandbytes as bnb
    import torch
    from datasets import load_from_disk
    from torch.profiler import ProfilerActivity, profile, record_function
    from unsloth import FastVisionModel

    import wandb

    from cartridges.training.collator import DeepSeekOCRDataCollator, to_conversation

    out = Path(PROFILE_ROOT) / run_name
    out.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(config.dataset_dir))["train"]
    examples = [dataset[i] for i in range(index, index + batch_size)]
    model, tokenizer = _load_base_and_lora(config)
    FastVisionModel.for_training(model)
    collator = DeepSeekOCRDataCollator(
        tokenizer=tokenizer,
        model=model,
        image_size=config.image_size,
        base_size=config.base_size,
        crop_mode=config.crop_mode,
        train_on_responses_only=config.train_on_responses_only,
    )
    # Trainer's optim="adamw_8bit" with weight_decay=0.001 (see _train()).
    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate, weight_decay=0.001
    )

    with _stage("collate", batch_size=batch_size):
        batch = collator([to_conversation(e) for e in examples])
    n_tokens = int(batch["input_ids"].numel())
    batch = _to_device(batch, "cuda")

    def step(timings: dict | None = None):
        def phase(name, fn):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with record_function(name):
                result = fn()
            torch.cuda.synchronize()
            if timings is not None:
                timings[name] = (time.perf_counter() - start) * 1000
            return result

        def forward():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(**batch).loss

        loss = phase("forward", forward)
        phase("backward", loss.backward)
        phase("optimizer", lambda: (optimizer.step(), optimizer.zero_grad(set_to_none=True)))
        return float(loss.detach())

    with _stage("warmup", steps=warmup_steps):
        for _ in range(warmup_steps):
            step()

    # Untraced timing first, so profiler overhead can be read off the traced numbers.
    untraced_ms: dict = {}
    step(untraced_ms)

    gpu_log = open(out / "gpu.csv", "w")
    sensor = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used",
         "--format=csv,nounits", "-lms", "50"],
        stdout=gpu_log,
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=200_000)
    traced_ms: dict = {}
    try:
        with _stage("profile_step", with_stack=with_stack), profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=with_stack,
            # export_stacks() comes back empty for CUDA metrics without verbose.
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        ) as prof:
            loss = step(traced_ms)
    finally:
        torch.cuda.memory._dump_snapshot(str(out / "memory.pickle"))
        torch.cuda.memory._record_memory_history(enabled=None)
        sensor.terminate()
        gpu_log.close()

    with _stage("export", out=out):
        prof.export_chrome_trace(str(out / "trace.json"))
        with open(out / "trace.json", "rb") as src, gzip.open(out / "trace.json.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        (out / "trace.json").unlink()
        if with_stack:
            prof.export_stacks(str(out / "stacks_cpu.folded"), "self_cpu_time_total")
            prof.export_stacks(str(out / "stacks_gpu.folded"), "self_cuda_time_total")
        averages = prof.key_averages()
        (out / "ops.txt").write_text(
            "== top 30 by self CUDA time ==\n"
            + averages.table(sort_by="self_cuda_time_total", row_limit=30)
            + "\n\n== top 30 by self CPU time ==\n"
            + averages.table(sort_by="self_cpu_time_total", row_limit=30)
        )
        cuda_api = count_cuda_api_calls([e.name for e in prof.events()])

    stages = {
        "run_name": run_name,
        "mode": "train-step",
        "batch_size": batch_size,
        "index": index,
        "n_tokens": n_tokens,
        "loss": loss,
        "phases_ms_traced": traced_ms,
        "phases_ms_untraced": untraced_ms,
        "step_ms_untraced": sum(untraced_ms.values()),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "cuda_api": cuda_api,
        "gpu": torch.cuda.get_device_name(),
    }
    (out / "stages.json").write_text(json.dumps(stages, indent=2))
    model_cache.commit()

    with _stage("log_summary", run_name=run_name):
        run = wandb.init(
            entity=WANDB_ENTITY, project=WANDB_PROJECT, group=PROFILE_GROUP,
            job_type="profile", name=run_name, tags=["profiling", "train-step"],
            config={"train_config": config.model_dump(), "batch_size": batch_size, "index": index},
        )
        flat = {f"untraced_ms/{k}": v for k, v in untraced_ms.items()}
        flat.update({f"traced_ms/{k}": v for k, v in traced_ms.items()})
        flat.update({f"cuda_api/{k}": v for k, v in cuda_api.items()})
        run.summary.update({**flat, "peak_mem_gb": stages["peak_mem_gb"], "n_tokens": n_tokens, "loss": loss})
        run.finish()

    logger.info("profile_train_step done: %s", stages)
    return stages


@app.local_entrypoint()
def run_profile_train(run_name: str, batch_size: int = 1, index: int = 0, with_stack: bool = True):
    import json

    stages = profile_train_step.remote(run_name, TrainConfig(), batch_size, index, 2, with_stack)
    print(json.dumps(stages, indent=2))
    print(f"fetch: uv run modal volume get cartridges-training-cache profiling/{run_name} profiles/")
