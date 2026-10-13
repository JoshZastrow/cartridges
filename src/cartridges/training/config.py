"""Configuration for the fine-tuning pipeline.

Fields that are filesystem paths on the Modal container are typed as `str`,
not `pathlib.Path` -- these configs get pickled locally (Python 3.13 on this
Mac) and unpickled remotely (the Modal image pins python_version="3.12").
`Path` objects don't round-trip across that version gap (Python 3.13 moved
concrete path classes into `pathlib._local`, which 3.12 doesn't have), but
plain `str` always does.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class BaseModelId(str, Enum):
    """Base vision-language models supported for fine-tuning."""

    DEEPSEEK_OCR_2 = "unsloth/DeepSeek-OCR-2"


class DownloadConfig(BaseModel):
    model_id: BaseModelId = BaseModelId.DEEPSEEK_OCR_2
    local_dir: str = "deepseek_ocr"


class TrainConfig(BaseModel):
    # Informational + used to label results -- must actually match whatever
    # model `download_model(DownloadConfig(model_id=...))` populated at
    # `base_model_dir`. Not derived automatically: renaming `base_model_dir`
    # to match `model_id` would orphan whatever's already been downloaded to
    # the current default path.
    model_id: BaseModelId = BaseModelId.DEEPSEEK_OCR_2
    base_model_dir: str = "/cache/deepseek_ocr"
    dataset_dir: str = "/data/hf_dataset"
    output_dir: str = "/cache/ocr_finetuned"

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4

    # HF TrainingArguments' scheduler knobs. Default ("linear", no kwargs)
    # decays learning_rate -> 0. To decay to a non-zero floor instead (e.g.
    # 2e-4 -> 1e-4), use "cosine_with_min_lr" with lr_scheduler_kwargs=
    # {"min_lr": 1e-4} -- supported by transformers>=4.44's get_scheduler.
    lr_scheduler_type: str = "linear"
    lr_scheduler_kwargs: dict = {}

    # Checkpoint every save_steps, pushed to the Hugging Face Hub repo below
    # (via TrainingArguments' push_to_hub/hub_strategy) so a run can be
    # inspected mid-training, not just at the final saved adapter. Switched
    # off W&B artifact storage entirely -- LoRA checkpoints there blew
    # through a 5GB account quota in one run.
    save_steps: int = 25
    save_total_limit: int = 3

    # Hugging Face Hub repo the adapter (and its checkpoints) get pushed to.
    # Private by default; the free plan has no meaningful storage cap for
    # LoRA-sized adapters, unlike W&B's artifact storage.
    hf_repo_id: str = "joshzastrow/deepseek-ocr2-lora"
    hf_private: bool = True

    # The vision encoder (deepencoderv2.py) only has two learned query
    # embeddings, sized for 768- or 1024-resolution views (144 or 256
    # "queries"). 640 (the reference Unsloth notebook's value, for a
    # different model) is architecturally unsupported here and raises
    # ValueError as soon as cropping actually kicks in.
    image_size: int = 768
    base_size: int = 1024
    crop_mode: bool = True
    train_on_responses_only: bool = True
    seed: int = 3407

    # model.infer()'s eval_mode generation caps at 8192 max_new_tokens
    # internally with no other early-stop guarantee; an undertrained/base
    # model that falls into a repetition loop can burn the full budget
    # (minutes per example). Capped to the longest labeled page instead:
    # 2,179 tokens (There_He_Is/83, data/train/quality/pages.csv) + ~17%
    # headroom, so no real page is truncated. Must stay <= fast_decode's
    # CudaGraphDecoder.MAX_CACHE_LEN minus the ~1,140-token prompt.
    max_new_tokens: int = 2560

    # W&B experiment tracking: `group` clusters every run from one sweep/
    # session together in the UI (https://docs.wandb.ai/models/runs/grouping);
    # `notes` and `tags` are freeform context written before a run executes --
    # set via CLI flags or a --config-file YAML (see finetune.py's
    # run_finetune entrypoint) rather than editing this file per run.
    group: str = "default"
    # Short W&B run name naming the experiment, e.g. "e2-train-eos-fix".
    # Empty lets W&B pick a random one.
    run_name: str = ""
    notes: str = ""
    tags: list[str] = []


class DecodingConfig(BaseModel):
    """Overrides for the generate() kwargs DeepSeek-OCR-2's model.infer()
    hardcodes (eval_mode: greedy, no_repeat_ngram_size=35, eos_token_id=
    tokenizer.eos_token_id). None keeps infer()'s own value.

    Defaults are the looping investigation's pick (E1b in
    docs/plans/loop-elimination.md): n-gram 10 + repetition penalty 1.05."""

    no_repeat_ngram_size: int | None = 10
    repetition_penalty: float | None = 1.05
    eos_token_id: int | None = None


class ModelSource(str, Enum):
    """Which model `evaluate()` should score."""

    BASE = "base"  # plain pretrained model, no adapter
    ARTIFACT = "artifact"  # a saved LoRA adapter, pulled from the HF Hub repo
    KFOLD_RETRAIN = "kfold_retrain"  # fresh LoRA per fold, trained on the spot


class EvalConfig(BaseModel):
    train_config: TrainConfig = TrainConfig()
    model_source: ModelSource = ModelSource.BASE

    # ARTIFACT only: a revision (git tag/branch/commit sha) in
    # train_config.hf_repo_id, e.g. "run-<wandb_run_id>", "date-2026-09-22",
    # or "main" for whatever's latest.
    hf_revision: str | None = None

    # BASE / ARTIFACT: which dataset split to score against.
    split: str = "test"

    # BASE / ARTIFACT: decode with fast_decode.CudaGraphDecoder instead of HF
    # generate (~12x faster per page; same quality, not bit-identical -- see
    # docs/plans/inference-speed.md). KFOLD_RETRAIN always uses HF generate.
    fast_decode: bool = True

    # KFOLD_RETRAIN only: folds over the "train" split.
    k: int = 5
    seed: int = 3407
