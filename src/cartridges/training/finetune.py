"""Modal-hosted LoRA fine-tuning for the OCR/transcription model.

Deps mirror processors/ocr.py's Modal app: declared in the Modal image, not
pyproject.toml/uv, since unsloth/bitsandbytes/triton/xformers/torchao need
CUDA+Linux and have no macOS wheels.

Deploy/run with:
uv run modal run src/cartridges/training/finetune.py
"""

from __future__ import annotations

import contextlib
import datetime
import logging
import os
import time

import modal

from cartridges.training.config import DecodingConfig, DownloadConfig, EvalConfig, ModelSource, TrainConfig

MINUTES = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cartridges.training")


@contextlib.contextmanager
def _stage(name: str, **fields):
    """Log a pipeline stage's start/success/failure + duration.

    Every stage in the pipeline (dataset load, model load, LoRA wrap, train,
    inference, artifact download, ...) goes through this, so a crash always
    names the exact stage it died in and how long it ran first, and a
    successful run leaves a clean start->duration trail to see where time
    actually goes.
    """
    context = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("[stage start] %s %s", name, context)
    start = time.monotonic()
    try:
        yield
    except Exception:
        logger.exception("[stage FAILED] %s after %.1fs", name, time.monotonic() - start)
        raise
    else:
        logger.info("[stage done] %s in %.1fs", name, time.monotonic() - start)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "unsloth",
        "unsloth_zoo",
        "bitsandbytes",
        "accelerate",
        "xformers==0.0.34",
        "peft",
        "trl==0.22.2",
        "triton",
        "torchao>=0.16.0",
        "transformers==4.56.2",
        "jiwer",
        "einops",
        "addict",
        "easydict",
        "sentencepiece",
        "protobuf",
        "huggingface_hub>=0.34.0",
        "hf_transfer",
        "wandb",
        "weave",
        # deepseek_ocr's trust_remote_code modeling file imports this at
        # module load time (unused by our .infer()/.generate() path, but
        # transformers' dynamic-module loader requires it be installed).
        "matplotlib",
    )
    # ships the dataset built by notebooks/001_build_ocr_dataset.py; rebuild
    # (redeploy) whenever data/train/hf_dataset changes. add_local_* must
    # come last in the chain (Modal requirement).
    .add_local_dir("data/train/hf_dataset", remote_path="/data/hf_dataset")
)

app = modal.App("cartridges-training", image=image)

# Create once with: modal secret create wandb-secret WANDB_API_KEY=<key>
# (key from https://wandb.ai/authorize). Only the GPU training functions use it.
wandb_secret = modal.Secret.from_name("wandb-secret")
hf_secret = modal.Secret.from_name("HF_TOKEN")
WANDB_ENTITY = "conle-ai"
WANDB_PROJECT = "cartridges"

model_cache = modal.Volume.from_name("cartridges-training-cache", create_if_missing=True)


@app.function(timeout=30 * MINUTES, volumes={"/cache": model_cache})
def download_model(config: DownloadConfig = DownloadConfig()) -> str:
    """Snapshot-download a base model into the Modal cache volume."""
    from huggingface_hub import snapshot_download

    local_dir = f"/cache/{config.local_dir}"
    with _stage("snapshot_download", model_id=config.model_id.value, local_dir=local_dir):
        snapshot_download(config.model_id.value, local_dir=local_dir)
        model_cache.commit()
    return local_dir


@app.function(timeout=10 * MINUTES, volumes={"/cache": model_cache})
def preview_collator_example(config: TrainConfig = TrainConfig(), index: int = 0) -> dict:
    """Run the collator on one training example, no GPU/model weights needed.

    Uses a dummy `dtype`-only stand-in for `model` (the collator only reads
    `model.dtype`) so this stays cheap: just the tokenizer + image tiling.
    """
    import sys
    import types
    from pathlib import Path

    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    from cartridges.training.collator import DeepSeekOCRDataCollator, to_conversation

    base_model_dir = Path(config.base_model_dir)
    sys.path.insert(0, str(base_model_dir.parent))

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    dummy_model = types.SimpleNamespace(dtype=torch.bfloat16)

    collator = DeepSeekOCRDataCollator(
        tokenizer=tokenizer,
        model=dummy_model,
        image_size=config.image_size,
        base_size=config.base_size,
        crop_mode=config.crop_mode,
        train_on_responses_only=config.train_on_responses_only,
    )

    dataset = load_from_disk(str(config.dataset_dir))["train"]
    example = dataset[index]
    conversation = to_conversation(example)
    processed = collator.process_single_sample(conversation["messages"])
    batch = collator([conversation])

    labels = batch["labels"][0]
    supervised_ids = [
        token_id
        for token_id, label in zip(batch["input_ids"][0].tolist(), labels.tolist())
        if label != -100
    ]

    return {
        "total_tokens": int(processed["input_ids"].shape[0]),
        "prompt_token_count": processed["prompt_token_count"],
        "num_image_tokens": int(processed["images_seq_mask"].sum()),
        "num_supervised_tokens": len(supervised_ids),
        "crop_ratio": processed["images_spatial_crop"].tolist(),
        "decoded_supervised_text": tokenizer.decode(supervised_ids),
    }


def _load_base_model(config: TrainConfig):
    """Load the base model + tokenizer, no LoRA."""
    import sys
    from pathlib import Path

    from transformers import AutoModel
    from unsloth import FastVisionModel

    # deepseek_ocr's own modeling module (imported by the collator) lives
    # inside the downloaded snapshot dir, e.g. /cache/deepseek_ocr/.
    base_model_dir = Path(config.base_model_dir)
    sys.path.insert(0, str(base_model_dir.parent))

    # from_pretrained(unsloth_force_compile=True) can sit compiling for
    # minutes on a cold container with zero output otherwise.
    with _stage("load_base_model", base_model_dir=base_model_dir, unsloth_force_compile=True):
        model, tokenizer = FastVisionModel.from_pretrained(
            str(base_model_dir),
            load_in_4bit=False,
            auto_model=AutoModel,
            trust_remote_code=True,
            unsloth_force_compile=True,
            use_gradient_checkpointing="unsloth",
        )
    # from_pretrained(local_dir) sets config._name_or_path to the local cache
    # path, which Trainer.push_to_hub()'s auto-generated model card embeds as
    # `base_model` metadata -- the Hub rejects that (not a real repo id).
    # Point it at the actual Hub id this snapshot came from.
    model.config._name_or_path = config.model_id.value

    # Fix A (docs/plans/inference-speed.md): transformers' DeepseekV2MoE.moe
    # checks `num_tokens == 0` on a CUDA tensor for each of 64 experts, a
    # blocking GPU->CPU sync per expert (~700 per forward). The patch reads
    # the per-expert counts once per layer; same math, identical outputs.
    # Class-level, so it covers training, eval and serving alike.
    from cartridges.training.fast_decode import _patch_moe

    with _stage("patch_moe_one_sync"):
        logger.info("patched MoE classes: %d", _patch_moe(model))
    return model, tokenizer


def _load_base_and_lora(config: TrainConfig):
    """Load the base model + tokenizer and wrap it with a fresh LoRA adapter."""
    from unsloth import FastVisionModel

    model, tokenizer = _load_base_model(config)

    with _stage("wrap_lora", lora_r=config.lora_r, lora_alpha=config.lora_alpha):
        model = FastVisionModel.get_peft_model(
            model,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            random_state=config.seed,
            use_rslora=False,
            loftq_config=None,
        )
    return model, tokenizer


def _init_train_run(config: TrainConfig) -> None:
    """Start the W&B run for a training job (full TrainConfig as config=,
    plus group/notes/tags). Callers start it as early as possible so the
    run's runtime and system (GPU) metrics cover model/data loading too."""
    import wandb

    with _stage("init_tracking", group=config.group):
        os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
        # Model checkpoints/adapters go to the HF Hub repo (push_to_hub in
        # TrainingArguments), not W&B artifacts -- W&B here is metrics/
        # config/CER only. (WANDB_LOG_MODEL was tried first; even just LoRA
        # adapters at every save_steps blew through a 5GB account quota in
        # one run.)
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group=config.group,
            job_type="finetune",
            name=config.run_name or None,
            notes=config.notes or None,
            tags=config.tags or None,
            config=config.dict(),
        )


def _train(
    model,
    tokenizer,
    config: TrainConfig,
    examples: list[dict],
    push_to_hub: bool = True,
    init_run: bool = True,
) -> dict:
    """Fine-tune `model` on `examples` ({"image", "text"} rows). Returns token stats.

    `push_to_hub=False` skips all Hub checkpoint/final pushes and tagging --
    used by `_evaluate_kfold()`, where each fold's LoRA is thrown away after
    scoring and would otherwise spam the repo with fold-scoped commits/tags.
    `init_run=False` means the caller already started the W&B run.
    """
    from unsloth import FastVisionModel, is_bf16_supported
    from transformers import Trainer, TrainingArguments

    from cartridges.training.collator import DeepSeekOCRDataCollator, to_conversation

    import wandb

    logger.info(
        "train start: n_examples=%d max_steps=%d lora_r=%d learning_rate=%s",
        len(examples), config.max_steps, config.lora_r, config.learning_rate,
    )

    # Explicitly init the run ourselves (full TrainConfig as config= -- see
    # https://docs.wandb.ai/models/track/config -- plus group/notes/tags per
    # https://docs.wandb.ai/models/runs/grouping) *before* trainer.train():
    # the Trainer's report_to="wandb" callback (below) then detects
    # wandb.run is already active and reuses it instead of creating its own
    # with just TrainingArguments as config. (weave.init() was tried here
    # too, but its project bootstrap check fails even for this account's
    # Admin role on the team -- plain wandb.init() works fine, so dropped
    # weave since it wasn't something actually needed.)
    if init_run:
        _init_train_run(config)

    if push_to_hub:
        with _stage("create_hf_repo", hf_repo_id=config.hf_repo_id):
            from huggingface_hub import HfApi

            HfApi().create_repo(config.hf_repo_id, private=config.hf_private, exist_ok=True)

    FastVisionModel.for_training(model)

    converted = [to_conversation(example) for example in examples]

    with _stage("build_collator"):
        data_collator = DeepSeekOCRDataCollator(
            tokenizer=tokenizer,
            model=model,
            image_size=config.image_size,
            base_size=config.base_size,
            crop_mode=config.crop_mode,
            train_on_responses_only=config.train_on_responses_only,
        )

    # token profiling: how many tokens this training set actually costs,
    # independent of whether training succeeds. Also a canary: if this stage
    # alone eats minutes, the collator's image tiling is the bottleneck, not
    # the actual GPU training step below.
    with _stage("token_profiling", n_examples=len(converted)):
        token_counts = [
            data_collator.process_single_sample(sample["messages"])["input_ids"].shape[0]
            for sample in converted
        ]

    with _stage("build_trainer"):
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=converted,
            args=TrainingArguments(
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                learning_rate=config.learning_rate,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.001,
                lr_scheduler_type=config.lr_scheduler_type,
                lr_scheduler_kwargs=config.lr_scheduler_kwargs,
                seed=config.seed,
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                output_dir="/tmp/outputs",
                report_to="wandb",
                dataloader_num_workers=2,
                remove_unused_columns=False,
                save_strategy="steps",
                save_steps=config.save_steps,
                save_total_limit=config.save_total_limit,
                # Skip gathering/writing optimizer + scheduler state on each
                # checkpoint -- adamw_8bit's optimizer state is what pushed a
                # 40GB A100 into OOM at a checkpoint boundary that plain
                # training never hit. Checkpoints are for artifact
                # versioning/inspection here, not exact-resume, so this
                # tradeoff is fine.
                save_only_model=True,
                # Push every saved checkpoint to its own subfolder in the HF
                # Hub repo (checkpoint-25/, checkpoint-50/, ...), not just
                # the final adapter -- the Hub-native equivalent of the old
                # WANDB_LOG_MODEL="checkpoint" behavior. Disabled entirely
                # for kfold retrains (push_to_hub=False) so those throwaway
                # folds don't spam the repo.
                push_to_hub=push_to_hub,
                hub_model_id=config.hf_repo_id if push_to_hub else None,
                hub_strategy="all_checkpoints",
                hub_private_repo=config.hf_private,
            ),
        )

    with _stage("trainer.train", max_steps=config.max_steps):
        trainer.train()

    import wandb
    from huggingface_hub import HfApi

    run = wandb.run  # the Trainer's report_to="wandb" callback already opened this

    if push_to_hub:
        with _stage("push_final_to_hub", hf_repo_id=config.hf_repo_id):
            # Puts the final adapter on `main` (not just its own checkpoint-N/
            # subfolder from hub_strategy="all_checkpoints" above), then tags
            # that commit by date/run-id -- same addressing the old W&B
            # artifact aliases gave, now as Hub git tags.
            trainer.push_to_hub()
            run_date = datetime.date.today().isoformat()
            api = HfApi()
            for tag in (f"date-{run_date}", f"run-{run.id if run else 'unknown'}"):
                api.create_tag(config.hf_repo_id, tag=tag, revision="main")

    logger.info("train done: wandb_run_url=%s", run.url if run else None)
    return {
        "total_tokens": sum(token_counts),
        "avg_tokens_per_example": sum(token_counts) / len(token_counts),
        "wandb_run_id": run.id if run else None,
        "wandb_run_name": run.name if run else None,
        "wandb_run_url": run.url if run else None,
        "hf_repo_id": config.hf_repo_id,
    }


def _truncate(text: str | None, n: int = 100) -> str:
    if text is None:
        return "<none>"
    text = text.replace("\n", "\\n")
    return text if len(text) <= n else text[:n] + "..."


class _NoTiming:
    """Stand-in for the profiler's timer/state that CudaGraphDecoder.generate()
    reports per-step timings to; evaluation doesn't need them."""

    def __init__(self):
        self.processor_s: list[float] = []
        self.processor_times: dict[str, list[float]] = {}

    def mark(self):
        pass


@contextlib.contextmanager
def _capped_generate(model, max_new_tokens: int, decoding: DecodingConfig | None = None, decoder=None):
    """model.infer() hardcodes max_new_tokens=8192 internally with no other
    reliable early-stop guarantee -- an undertrained/repetitive model can
    burn the full budget per example (minutes). infer() doesn't expose this
    as a parameter, so monkeypatch generate() (restored on exit) to clamp
    whatever it's called with. The same hook applies any `decoding`
    overrides over infer()'s hardcoded kwargs, and, with `decoder` (a
    fast_decode.CudaGraphDecoder), replaces HF generate entirely.

    Patches the model `infer()` actually runs on: on a PeftModel that's the
    inner model (get_base_model()), not the wrapper -- patching the wrapper
    silently left adapter evals uncapped."""
    target = model.get_base_model() if hasattr(model, "get_base_model") else model
    original_generate = target.generate
    overrides = decoding.model_dump(exclude_none=True) if decoding else {}
    if decoder is not None and "repetition_penalty" in overrides:
        raise ValueError("fast_decode supports eos_token_id/no_repeat_ngram_size overrides, not repetition_penalty")

    def capped(*args, **kwargs):
        kwargs["max_new_tokens"] = min(kwargs.get("max_new_tokens", max_new_tokens), max_new_tokens)
        kwargs.update(overrides)
        if decoder is not None:
            return decoder.generate(*args, timer=_NoTiming(), state=_NoTiming(), **kwargs)
        return original_generate(*args, **kwargs)

    target.generate = capped
    try:
        yield
    finally:
        target.generate = original_generate


def _run_inference(
    model,
    tokenizer,
    examples: list[dict],
    config: TrainConfig,
    decoding: DecodingConfig | None = None,
    fast_decode: bool = False,
) -> list[dict]:
    """Run `model.infer` over `examples`; logs per-example progress/score/errors
    as it goes (so a crash mid-run still leaves useful Modal logs), and returns
    [{"index", "reference", "hypothesis", "cer", "status", "elapsed_s"}, ...].

    `fast_decode` swaps HF generate for fast_decode.CudaGraphDecoder (one host
    sync per MoE layer instead of one per expert, CUDA-graph decode step,
    batched routed experts; LoRA stays unmerged). Same greedy/EOS/no-repeat-
    ngram semantics, ~12x faster per page, but not bit-identical to HF
    generate (docs/plans/inference-speed.md).
    """
    import tempfile
    import time
    from pathlib import Path

    import jiwer
    from unsloth import FastVisionModel

    from cartridges.training.collator import FREE_OCR_PROMPT

    n = len(examples)
    with _stage(
        "run_inference", n_examples=n, image_size=config.image_size,
        base_size=config.base_size, crop_mode=config.crop_mode,
        max_new_tokens=config.max_new_tokens, decoding=decoding,
    ):
        FastVisionModel.for_inference(model)

        decoder = None
        if fast_decode:
            from cartridges.training.fast_decode import CudaGraphDecoder, _moe_modules, _patch_moe

            with _stage("setup_fast_decode"):
                _patch_moe(model)
                decoder = CudaGraphDecoder(model.get_base_model() if hasattr(model, "get_base_model") else model)
                for moe in _moe_modules(model):
                    moe._batched = True

        predictions = []
        with tempfile.TemporaryDirectory() as tmpdir, _capped_generate(
            model, config.max_new_tokens, decoding, decoder
        ):
            for i, example in enumerate(examples):
                start = time.monotonic()
                reference = example["text"]
                image_path = Path(tmpdir) / f"{i}.png"
                example["image"].save(image_path)

                hypothesis = None
                cer = None
                status = "ok"
                try:
                    hypothesis = model.infer(
                        tokenizer,
                        prompt=FREE_OCR_PROMPT,
                        image_file=str(image_path),
                        output_path=tmpdir,
                        base_size=config.base_size,
                        image_size=config.image_size,
                        crop_mode=config.crop_mode,
                        save_results=False,
                        test_compress=False,
                        # without this, infer() only streams tokens to stdout
                        # and returns None -- this was the root cause of the
                        # earlier "unrelated text" baseline output: that was
                        # streamed stdout, not the (None) return value.
                        eval_mode=True,
                    )
                    cer = jiwer.cer(reference, hypothesis)
                except Exception:
                    status = "error"
                    logger.exception("inference failed on example %d/%d", i + 1, n)

                elapsed = time.monotonic() - start
                logger.info(
                    "example %d/%d status=%s cer=%s elapsed=%.1fs ref=%r pred=%r",
                    i + 1, n, status,
                    f"{cer:.2%}" if cer is not None else "n/a",
                    elapsed,
                    _truncate(reference),
                    _truncate(hypothesis),
                )
                predictions.append(
                    {
                        "index": i,
                        "reference": reference,
                        "hypothesis": hypothesis,
                        "cer": cer,
                        "status": status,
                        "elapsed_s": elapsed,
                    }
                )

        from cartridges.training.metrics import corpus_cer

        n_errors = sum(p["hypothesis"] is None for p in predictions)
        cer = corpus_cer([p["reference"] for p in predictions], [p["hypothesis"] for p in predictions])
        logger.info(
            "inference summary: n_examples=%d n_errors=%d corpus_cer=%s n_blank_refs=%d",
            n, n_errors, f"{cer['corpus_cer']:.2%}" if cer["corpus_cer"] is not None else "n/a",
            cer["n_blank_refs"],
        )
    return predictions


def _load_artifact_model(config: EvalConfig):
    """Load the base model + tokenizer, then attach a saved LoRA adapter pulled
    from the HF Hub repo (`peft.PeftModel.from_pretrained(repo_id, revision=...)`
    -- symmetric to `trainer.push_to_hub()` in `_train()`)."""
    from peft import PeftModel

    model, tokenizer = _load_base_model(config.train_config)

    with _stage(
        "attach_adapter", hf_repo_id=config.train_config.hf_repo_id, hf_revision=config.hf_revision
    ):
        model = PeftModel.from_pretrained(
            model, config.train_config.hf_repo_id, revision=config.hf_revision or "main"
        )

    return model, tokenizer


@app.function(
    # Tried L4 (24GB, cheaper) since GPU memory usage was low (~13.7GB) --
    # but L4's memory bandwidth is ~1/5th of A100's, and both the training
    # forward pass and (especially) autoregressive generation are bandwidth-
    # bound, not just VRAM-bound. Saw a single eval example take 20 minutes
    # on L4 vs ~1 minute on A100. Reverted.
    # 80GB variant: plain 40GB A100 ("A100") OOM'd at a checkpoint-save
    # boundary once save_strategy="steps" started actually saving mid-run.
    gpu="A100-80GB",
    timeout=4 * 60 * MINUTES,
    volumes={"/cache": model_cache},
    secrets=[wandb_secret, hf_secret],
)
def finetune(config: TrainConfig = TrainConfig()) -> dict:
    """LoRA fine-tune the base model, save the adapter locally (Modal volume
    cache) and push it to the Hub -- see `_train()`'s push_final_to_hub stage
    for the repo/tags.

    Returns {"output_dir", "hf_repo_id", "wandb_run_url", ...run_info} so
    callers (experiment sweeps) can chain straight into `evaluate()` with
    `model_source=ARTIFACT` without a manual lookup.
    """
    from datasets import load_from_disk

    import wandb

    _init_train_run(config)

    with _stage("load_dataset", dataset_dir=config.dataset_dir):
        dataset = list(load_from_disk(str(config.dataset_dir))["train"])

    model, tokenizer = _load_base_and_lora(config)
    run_info = _train(model, tokenizer, config, dataset, init_run=False)

    with _stage("save_model", output_dir=config.output_dir):
        model.save_pretrained(str(config.output_dir))
        tokenizer.save_pretrained(str(config.output_dir))
        model_cache.commit()

    wandb.finish()

    logger.info("finetune done: output_dir=%s hf_repo_id=%s", config.output_dir, config.hf_repo_id)
    return {
        "output_dir": str(config.output_dir),
        **run_info,
    }


def _evaluate_kfold(config: EvalConfig) -> list[dict]:
    """K-fold cross-validate over the "train" split: fresh LoRA per fold."""
    import gc
    import random

    import torch
    from datasets import load_from_disk

    train_config = config.train_config
    with _stage("load_dataset", dataset_dir=train_config.dataset_dir, split="train"):
        dataset = list(load_from_disk(str(train_config.dataset_dir))["train"])
    indices = list(range(len(dataset)))
    random.Random(config.seed).shuffle(indices)
    folds = [indices[i :: config.k] for i in range(config.k)]
    logger.info(
        "kfold start: model_id=%s k=%d n_examples=%d max_steps=%d",
        train_config.model_id.value, config.k, len(dataset), train_config.max_steps,
    )

    results = []
    for fold_idx, eval_indices in enumerate(folds):
        eval_set = set(eval_indices)
        train_examples = [dataset[i] for i in indices if i not in eval_set]
        eval_examples = [dataset[i] for i in eval_indices]
        logger.info(
            "fold %d/%d start: train_size=%d eval_size=%d",
            fold_idx + 1, config.k, len(train_examples), len(eval_examples),
        )

        model, tokenizer = _load_base_and_lora(train_config)
        token_stats = _train(model, tokenizer, train_config, train_examples, push_to_hub=False)
        predictions = _run_inference(model, tokenizer, eval_examples, train_config)
        logger.info("fold %d/%d done", fold_idx + 1, config.k)

        results.append(
            {
                "fold": fold_idx,
                "model_id": train_config.model_id.value,
                "model_source": config.model_source.value,
                "hf_revision": None,
                "train_size": len(train_examples),
                "eval_size": len(eval_examples),
                "predictions": predictions,
                **token_stats,
            }
        )

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    return results


@app.function(
    # Reverted to A100 -- see finetune()'s comment. Generation is memory-
    # bandwidth-bound, and L4's bandwidth is far too low for this workload.
    gpu="A100",
    timeout=8 * 60 * MINUTES,
    volumes={"/cache": model_cache},
    secrets=[wandb_secret, hf_secret],
)
def evaluate(config: EvalConfig = EvalConfig()) -> list[dict]:
    """Standard eval entrypoint. `config.model_source` picks what gets scored:

    - BASE: the plain pretrained model, no adapter -- the baseline to beat.
    - ARTIFACT: a saved LoRA adapter pulled from the HF Hub repo
      (`config.train_config.hf_repo_id` at revision `config.hf_revision`,
      e.g. "run-<wandb_run_id>" or "date-2026-09-22").
    - KFOLD_RETRAIN: fresh LoRA trained per fold over the "train" split,
      for iterating on hyperparameters before committing to a real run.

    Always just loads a model, loads data, and runs inference -- returns raw
    (reference, hypothesis) pairs per "fold" (a single one for BASE/ARTIFACT).
    CER (or any other metric) is computed locally in the notebook.
    """
    logger.info(
        "evaluate start: model_id=%s model_source=%s hf_revision=%s split=%s",
        config.train_config.model_id.value, config.model_source.value, config.hf_revision, config.split,
    )

    if config.model_source == ModelSource.KFOLD_RETRAIN:
        return _evaluate_kfold(config)

    from datasets import load_from_disk

    with _stage("load_dataset", dataset_dir=config.train_config.dataset_dir, split=config.split):
        dataset = list(load_from_disk(str(config.train_config.dataset_dir))[config.split])

    # If hf_revision follows the "run-<wandb_run_id>" tag convention _train()
    # creates, that's the wandb run that produced this adapter -- recover it
    # so eval_cer can be written back onto that run's own summary below.
    # (No such recovery for a "date-..." tag, "main", or a checkpoint
    # subfolder -- the CER writeback is then simply skipped.)
    wandb_run_id = None
    if config.model_source == ModelSource.BASE:
        model, tokenizer = _load_base_model(config.train_config)
    else:
        if not config.hf_revision:
            raise ValueError("hf_revision is required when model_source='artifact'")
        model, tokenizer = _load_artifact_model(config)
        if config.hf_revision.startswith("run-"):
            wandb_run_id = config.hf_revision.removeprefix("run-")

    predictions = _run_inference(model, tokenizer, dataset, config.train_config, fast_decode=config.fast_decode)

    if wandb_run_id:
        # Write eval_cer onto the training run's own summary, alongside its
        # training metrics -- https://docs.wandb.ai/models/track/log/log-summary
        # -- so the run that produced an artifact also shows how it scored,
        # without needing to cross-reference a separate eval run.
        scored = [p["cer"] for p in predictions if p["cer"] is not None]
        if scored:
            mean_cer = sum(scored) / len(scored)
            with _stage("log_eval_summary", wandb_run_id=wandb_run_id, eval_cer=mean_cer):
                import wandb

                api = wandb.Api()
                api_run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{wandb_run_id}")
                api_run.summary["eval_cer"] = mean_cer
                api_run.summary[f"eval_cer_{config.split}"] = mean_cer
                api_run.summary["eval_n"] = len(scored)
                from cartridges.training.metrics import corpus_cer

                api_run.summary["eval_corpus_cer"] = corpus_cer(
                    [p["reference"] for p in predictions], [p["hypothesis"] for p in predictions]
                )["corpus_cer"]
                api_run.summary.update()

    return [
        {
            "fold": None,
            "model_id": config.train_config.model_id.value,
            "model_source": config.model_source.value,
            "hf_repo_id": config.train_config.hf_repo_id,
            "hf_revision": config.hf_revision,
            "train_size": None,
            "eval_size": len(predictions),
            "predictions": predictions,
            "total_tokens": None,
            "avg_tokens_per_example": None,
            "wandb_run_id": wandb_run_id,
        }
    ]


# --- looping investigation (docs/plans/loop-elimination.md) ---------------

LOOP_GROUP = "eos-loop-debug"


@app.function(
    gpu="A100", timeout=4 * 60 * MINUTES, volumes={"/cache": model_cache}, secrets=[hf_secret, wandb_secret]
)
def loop_eval_shard(
    config: EvalConfig,
    decoding: DecodingConfig,
    items: list[tuple[str, int]],
    wandb_run_id: str | None = None,
    shard: int = 0,
) -> list[dict]:
    """Transcribe one shard of (split, index) pages with the model `config` names.

    With `wandb_run_id`, joins the coordinator's run as a W&B shared-mode
    worker (https://docs.wandb.ai/models/track/log/distributed-training) so
    this container's GPU system metrics are recorded under label shard-N."""
    from datasets import load_from_disk

    if wandb_run_id:
        import wandb

        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            id=wandb_run_id,
            settings=wandb.Settings(
                mode="shared",
                x_primary=False,
                x_label=f"shard-{shard}",
                x_update_finish_state=False,
                # default is ~15s, too coarse to see GPU utilization per page
                x_stats_sampling_interval=5.0,
            ),
        )

    dataset = load_from_disk(str(config.train_config.dataset_dir))
    examples = [dataset[split][index] for split, index in items]
    if config.model_source == ModelSource.BASE:
        model, tokenizer = _load_base_model(config.train_config)
    else:
        model, tokenizer = _load_artifact_model(config)

    predictions = _run_inference(model, tokenizer, examples, config.train_config, decoding, config.fast_decode)
    import torch

    # Only the primary (CPU coordinator) uploads hardware metadata in W&B
    # shared mode, so report the GPU back for the run summary.
    gpu_name = torch.cuda.get_device_name()
    for prediction, (split, index), example in zip(predictions, items, examples):
        prediction.update(
            split=split, index=index, journal=example["journal"], page=example["page"], gpu=gpu_name
        )
    if wandb_run_id:
        wandb.finish()
    return predictions


@app.function(timeout=6 * 60 * MINUTES, secrets=[wandb_secret])
def loop_eval(
    config: EvalConfig,
    items: list[tuple[str, int]],
    decoding: DecodingConfig = DecodingConfig(),
    run_name: str = "",
    notes: str = "",
    tags: list[str] = [],
    shards: int = 4,
) -> dict:
    """Score `loop_rate` (see training/loops.py) over `items`, fanned out
    across `shards` GPU containers, and log one W&B run to LOOP_GROUP. The
    W&B table holds page ids and scores only, never transcription text."""
    import wandb

    import json

    from cartridges.training.loops import ends_in_loop, loop_rate
    from cartridges.training.metrics import corpus_cer

    # Init before inference so the run shows as "running" in the group while
    # the shards work, instead of appearing only once everything is done.
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=LOOP_GROUP,
        job_type="loop-eval",
        name=run_name or None,
        notes=notes or None,
        tags=["looping", *tags],
        config={"eval": config.model_dump(mode="json"), "decoding": decoding.model_dump(), "shards": shards},
        settings=wandb.Settings(mode="shared", x_primary=True, x_label="coordinator"),
    )
    logger.info("loop_eval wandb run: %s", run.url)

    chunks = [items[i::shards] for i in range(shards) if items[i::shards]]
    shard_args = [(config, decoding, chunk, run.id, i) for i, chunk in enumerate(chunks)]
    predictions = [p for chunk in loop_eval_shard.starmap(shard_args) for p in chunk]
    for p in predictions:
        p["looped"] = ends_in_loop(p["hypothesis"]) and not ends_in_loop(p["reference"])

    scored = [p for p in predictions if p["hypothesis"] is not None]
    rate = loop_rate([p["hypothesis"] for p in predictions], [p["reference"] for p in predictions])
    cer = corpus_cer([p["reference"] for p in predictions], [p["hypothesis"] for p in predictions])
    summary = {
        "loop_rate": rate,
        "n_looped": sum(p["looped"] for p in scored),
        "n_pages": len(scored),
        "n_errors": len(predictions) - len(scored),
        "corpus_cer": cer["corpus_cer"],
        "n_blank_refs": cer["n_blank_refs"],
        "shard_gpus": sorted({p["gpu"] for p in predictions}),
    }

    with _stage("log_loop_eval", run_name=run_name, **summary):
        manifest = wandb.Artifact("loop-eval-items", type="dataset", description="(split, index) page ids")
        with manifest.new_file("items.json") as f:
            json.dump(items, f)
        run.use_artifact(manifest)
        table = wandb.Table(columns=["split", "index", "journal", "page", "looped", "cer", "hyp_words", "ref_words"])
        for p in sorted(predictions, key=lambda p: (p["split"], p["index"])):
            table.add_data(
                p["split"], p["index"], p["journal"], p["page"], p["looped"], p["cer"],
                len((p["hypothesis"] or "").split()), len(p["reference"].split()),
            )
        run.log({"pages": table, **{k: v for k, v in summary.items() if isinstance(v, (int, float))}})
        run.summary.update(summary)
        url = run.url
        run.finish()

    return {**summary, "run_name": run_name, "wandb_url": url, "predictions": predictions}


# --- local entrypoints -----------------------------------------------------
#
# Long-running `.remote()` calls held open a single blocking connection for
# too long in this environment and eventually died with
# "StreamTerminatedError: Connection lost" partway through, even with retries.
# `modal run --detach <this file>::<entrypoint> -- <flags>` decouples the
# app's lifetime from the local client entirely: the entrypoint just submits
# a `.spawn()` (a few seconds of connection) and prints a call id; poll for
# the result later (see scripts/poll_call.py-style short reconnects) instead
# of blocking on one long-lived stream.


@app.local_entrypoint()
def run_evaluate(
    model_source: str = "base",
    hf_revision: str = "",
    split: str = "test",
    k: int = 5,
    group: str = "default",
):
    config = EvalConfig(
        train_config=TrainConfig(group=group),
        model_source=ModelSource(model_source),
        hf_revision=hf_revision or None,
        split=split,
        k=k,
    )
    call = evaluate.spawn(config)
    print(f"CALL_ID {call.object_id}")


@app.local_entrypoint()
def run_loop_eval(
    run_name: str,
    items: str = "all",
    model_source: str = "base",
    hf_revision: str = "",
    no_repeat_ngram_size: int | None = None,
    repetition_penalty: float | None = None,
    eos_token_id: int | None = None,
    notes: str = "",
    tags: str = "",
    shards: int = 4,
    fast_decode: bool = True,
):
    """Spawn a `loop_eval`. `items` is "all", a split name, or a JSONL
    manifest of {"split", "index"} rows (data/train/loop_eval/manifest.jsonl).

    uv run modal run --detach -m cartridges.training.finetune::run_loop_eval \\
        --run-name baseline-base --model-source base
    """
    import json

    from datasets import load_from_disk

    if items.endswith(".jsonl"):
        with open(items) as f:
            pairs = [(row["split"], row["index"]) for row in map(json.loads, f)]
    else:
        dataset = load_from_disk("data/train/hf_dataset")
        splits = list(dataset) if items == "all" else [items]
        pairs = [(split, i) for split in splits for i in range(len(dataset[split]))]

    config = EvalConfig(
        model_source=ModelSource(model_source), hf_revision=hf_revision or None, fast_decode=fast_decode
    )
    decoding = DecodingConfig(
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
    )
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    print(f"loop_eval: run_name={run_name} n_items={len(pairs)} decoding={decoding.model_dump(exclude_none=True)}")
    call = loop_eval.spawn(config, pairs, decoding, run_name, notes, tag_list, shards)
    print(f"CALL_ID {call.object_id}")


@app.local_entrypoint()
def run_finetune(
    config_file: str = "",
    max_steps: int | None = None,
    learning_rate: float | None = None,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    per_device_train_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    warmup_steps: int | None = None,
    group: str | None = None,
    run_name: str | None = None,
    notes: str | None = None,
    tags: str = "",
):
    """Launch a fine-tune run.

    Config precedence: TrainConfig defaults < --config-file YAML < explicit
    CLI flags. --config-file points at a YAML file you edit per run/sweep to
    write notes/hyperparameters before executing, instead of a long CLI
    invocation each time -- e.g.:

        model_id: unsloth/DeepSeek-OCR-2
        max_steps: 100
        learning_rate: 0.0002
        group: lr-sweep-2026-09-21
        notes: "Trying a lower LR after the 60-step run overfit."
        tags: [lr-sweep, lora16]

    uv run modal run -m cartridges.training.finetune::run_finetune \\
        --config-file configs/exp1.yaml --notes "quick check"
    """
    overrides: dict = {}
    if config_file:
        import yaml

        with open(config_file) as f:
            overrides = yaml.safe_load(f) or {}

    cli_values = {
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "group": group,
        "run_name": run_name,
        "notes": notes,
    }
    overrides.update({k: v for k, v in cli_values.items() if v is not None})
    if tags:
        overrides["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

    config = TrainConfig(**overrides)
    print(f"config: {config.dict()}")
    call = finetune.spawn(config)
    print(f"CALL_ID {call.object_id}")


@app.function(secrets=[wandb_secret])
def wandb_whoami() -> dict:
    """Diagnostic: which W&B account does wandb-secret's key actually
    authenticate as, and which entities can it write to."""
    import wandb

    api = wandb.Api()
    viewer = api.viewer
    entities = []
    try:
        entities = [e["node"]["name"] for e in viewer["teams"]["edges"]]
    except Exception as e:
        entities = [f"<could not parse teams: {e}>"]
    return {"username": viewer.username, "entities": entities}


@app.local_entrypoint()
def check_wandb():
    call = wandb_whoami.spawn()
    print(f"CALL_ID {call.object_id}")


@app.function(secrets=[wandb_secret])
def wandb_check_project(entity: str, project: str = "cartridges") -> dict:
    """Diagnostic: can this key read `entity/project`, and separately, can
    it actually create/write a new project there (what finetune() needs)?"""
    import wandb

    api = wandb.Api()
    result = {"entity": entity, "project": project}

    try:
        list(api.projects(entity=entity, per_page=1))
        result["can_list_projects"] = True
    except Exception as e:
        result["can_list_projects"] = f"{type(e).__name__}: {e}"

    try:
        run = wandb.init(entity=entity, project=project, job_type="diagnostic")
        result["can_create_run"] = True
        run.finish()
    except Exception as e:
        result["can_create_run"] = f"{type(e).__name__}: {e}"

    return result


@app.local_entrypoint()
def check_wandb_project(entity: str, project: str = "cartridges"):
    call = wandb_check_project.spawn(entity, project)
    print(f"CALL_ID {call.object_id}")


@app.function(volumes={"/cache": model_cache})
def check_tokenizer_ids() -> dict:
    """Diagnostic: does this tokenizer's pad_token_id collide with
    eos_token_id? If so, the collator's value-based label masking
    (`labels[labels == pad_token_id] = -100`) silently erases EOS
    supervision too, since it can't tell "real EOS" from "padding" by
    value alone when they're the same id."""
    import sys
    from pathlib import Path

    from transformers import AutoTokenizer

    base_model_dir = Path("/cache/deepseek_ocr")
    sys.path.insert(0, str(base_model_dir.parent))
    tok = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    return {
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
        "pad_token": tok.pad_token,
        "pad_token_id": tok.pad_token_id,
        "collide": tok.eos_token_id == tok.pad_token_id,
    }


@app.local_entrypoint()
def check_tokenizer():
    call = check_tokenizer_ids.spawn()
    print(f"CALL_ID {call.object_id}")


@app.function(gpu="L4", volumes={"/cache": model_cache}, timeout=10 * MINUTES)
def check_unsloth_tokenizer_ids() -> dict:
    """Same check, but on the tokenizer as it's actually used in our
    pipeline -- loaded via unsloth.FastVisionModel.from_pretrained, not a
    plain AutoTokenizer. Unsloth is known to override pad_token_id to equal
    eos_token_id for models lacking a pad token."""
    tokenizer = _load_base_model(TrainConfig())[1]
    return {
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "collide": tokenizer.eos_token_id == tokenizer.pad_token_id,
    }


@app.local_entrypoint()
def check_unsloth_tokenizer():
    call = check_unsloth_tokenizer_ids.spawn()
    print(f"CALL_ID {call.object_id}")


@app.function(gpu="L4", volumes={"/cache": model_cache}, timeout=15 * MINUTES)
def eos_audit(config: TrainConfig = TrainConfig()) -> dict:
    """Diagnostic for the looping investigation: does EOS survive into the
    training labels, and does inference stop on the same EOS id the model was
    pre-trained with? Collates a padded batch of two training examples with
    the Unsloth-loaded tokenizer (the one training actually uses)."""
    from datasets import load_from_disk

    from cartridges.training.collator import DeepSeekOCRDataCollator, to_conversation

    model, tokenizer = _load_base_model(config)
    collator = DeepSeekOCRDataCollator(
        tokenizer=tokenizer,
        model=model,
        image_size=config.image_size,
        base_size=config.base_size,
        crop_mode=config.crop_mode,
        train_on_responses_only=config.train_on_responses_only,
    )
    dataset = load_from_disk(str(config.dataset_dir))["train"]
    batch = collator([to_conversation(dataset[i]) for i in (0, 1)])

    samples = []
    for row in range(2):
        input_ids = batch["input_ids"][row].tolist()
        labels = batch["labels"][row].tolist()
        n_real = int(batch["attention_mask"][row].sum())
        eos_positions = [i for i, t in enumerate(input_ids) if t == tokenizer.eos_token_id]
        supervised = [i for i, label in enumerate(labels) if label != -100]
        samples.append(
            {
                "n_tokens": len(input_ids),
                "n_attended": n_real,
                "eos_positions_in_input": eos_positions[:5],
                "eos_supervised": any(labels[i] == tokenizer.eos_token_id for i in eos_positions),
                "last_supervised_token": tokenizer.decode([input_ids[supervised[-1]]]) if supervised else None,
                "tail_tokens": tokenizer.convert_ids_to_tokens(input_ids[-4:]),
            }
        )

    return {
        "tokenizer_eos": (tokenizer.eos_token, tokenizer.eos_token_id),
        "tokenizer_pad": (tokenizer.pad_token, tokenizer.pad_token_id),
        "pad_equals_eos": tokenizer.pad_token_id == tokenizer.eos_token_id,
        "model_config_eos_token_id": model.config.eos_token_id,
        "generation_config_eos_token_id": getattr(model.generation_config, "eos_token_id", None),
        "encode_eos_text": tokenizer.encode(f" {tokenizer.eos_token}", add_special_tokens=False),
        "samples": samples,
    }


@app.local_entrypoint()
def run_eos_audit():
    print(eos_audit.remote())
