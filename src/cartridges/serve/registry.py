"""Model registry: resolve, verify, and promote OCR adapters.

The registry is the HF Hub repo that training pushes to (TrainConfig.hf_repo_id).
- resolve_revision(): a tag or branch -> the commit SHA. "base" stays "base"
  (the base model, no adapter).
- check_promotion(): eval predictions -> pass or fail, with the reasons.
- tag_verified(): adds the tag verified-<date>-<sha7> and moves the tag prod.

The record of the deployed model is in a Modal Dict, so a client can read
the served SHA without a request to the GPU server (no cold start).
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel

from cartridges.training.config import TrainConfig
from cartridges.training.loops import loop_rate

BASE = "base"
HF_REPO_ID = TrainConfig().hf_repo_id
PROD_TAG = "prod"
DEPLOY_DICT = "cartridges-ocr-deploy"


def resolve_revision(revision: str, repo_id: str = HF_REPO_ID) -> str:
    if revision == BASE:
        return BASE
    from huggingface_hub import HfApi

    return HfApi().model_info(repo_id, revision=revision).sha


class PromotionResult(BaseModel):
    n: int
    n_errors: int
    mean_cer: float | None
    loop_rate: float
    passed: bool
    reasons: list[str]


def check_promotion(
    predictions: list[dict],
    *,
    max_cer: float,
    max_loop_rate: float,
    max_error_rate: float = 0.0,
) -> PromotionResult:
    """predictions: the rows of finetune.evaluate() (keys: reference,
    hypothesis, cer; cer is None for a failed example)."""
    n = len(predictions)
    scored = [p["cer"] for p in predictions if p["cer"] is not None]
    n_errors = n - len(scored)
    mean_cer = sum(scored) / len(scored) if scored else None
    rate = loop_rate([p["hypothesis"] for p in predictions], [p["reference"] for p in predictions])

    reasons = []
    if n == 0:
        reasons.append("no predictions")
    if mean_cer is None or mean_cer > max_cer:
        reasons.append(f"mean_cer {mean_cer} > {max_cer}")
    if rate > max_loop_rate:
        reasons.append(f"loop_rate {rate:.4f} > {max_loop_rate}")
    if n and n_errors / n > max_error_rate:
        reasons.append(f"error rate {n_errors}/{n} > {max_error_rate}")
    return PromotionResult(
        n=n, n_errors=n_errors, mean_cer=mean_cer, loop_rate=rate,
        passed=not reasons, reasons=reasons,
    )


def tag_verified(sha: str, repo_id: str = HF_REPO_ID, today: datetime.date | None = None) -> list[str]:
    """Add verified-<date>-<sha7> and move prod to `sha`. Hub tags do not
    move, so delete prod first, then make it again."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError

    api = HfApi()
    verified = f"verified-{(today or datetime.date.today()).isoformat()}-{sha[:7]}"
    api.create_tag(repo_id, tag=verified, revision=sha, exist_ok=True)
    try:
        api.delete_tag(repo_id, tag=PROD_TAG)
    except RevisionNotFoundError:
        pass
    api.create_tag(repo_id, tag=PROD_TAG, revision=sha)
    return [verified, PROD_TAG]


def record_deploy(revision: str, model_sha: str) -> dict:
    import modal

    record = {
        "revision": revision,
        "model_sha": model_sha,
        "deployed_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    modal.Dict.from_name(DEPLOY_DICT, create_if_missing=True)["serve"] = record
    return record


def deployed() -> dict | None:
    """The record that record_deploy() wrote, or None."""
    import modal

    return modal.Dict.from_name(DEPLOY_DICT, create_if_missing=True).get("serve")
